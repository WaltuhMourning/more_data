import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os

# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------
@st.cache_data
def load_data(base_path: str):
    """
    Load all relevant CSV files for a given chamber (Senate or House).
    Returns a dict of DataFrames:
        {
          "messages_per_day": pd.DataFrame,
          "words_per_day": pd.DataFrame,
          "user_stats": pd.DataFrame,
          "user_word_frequencies": pd.DataFrame,
          "word_frequencies": pd.DataFrame
        }
    """
    data = {}
    files = {
        "messages_per_day": "messages_per_day.csv",
        "words_per_day": "words_per_day.csv",
        "user_stats": "user_stats.csv",
        "user_word_frequencies": "user_word_frequencies.csv",
        "word_frequencies": "word_frequencies.csv",
    }

    for key, filename in files.items():
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Rename the first column to 'date' if needed (for messages/words per day)
            if key in ["messages_per_day", "words_per_day"]:
                if df.columns[0].startswith("Unnamed") or df.columns[0].strip() == "":
                    df.rename(columns={df.columns[0]: "date"}, inplace=True)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors='coerce')
            data[key] = df
        else:
            data[key] = pd.DataFrame()

    return data


def generate_wordcloud_from_frequencies(freq_dict, max_words=300, width=800, height=400):
    """
    Given a dict {word -> frequency}, generate a Word Cloud image.
    """
    wc = WordCloud(width=width, height=height, background_color='white')
    wc.generate_from_frequencies(freq_dict)
    return wc


def get_overall_stats(df_user_stats: pd.DataFrame):
    """
    Return a dict with overall stats:
      - total messages
      - total words
      - combined avg messages/day (all users)
      - combined avg words/day (all users)
    """
    if df_user_stats.empty:
        return {}
    total_messages = df_user_stats['total_messages'].sum()
    total_words = df_user_stats['total_words'].sum()

    # renamed for clarity:
    combined_avg_msgs_day = df_user_stats['avg_messages_per_day'].sum()
    combined_avg_words_day = df_user_stats['avg_words_per_day'].sum()

    stats = {
        'Total Messages': total_messages,
        'Total Words': total_words,
        'Combined Avg. Messages/Day (All Users)': combined_avg_msgs_day,
        'Combined Avg. Words/Day (All Users)': combined_avg_words_day
    }
    return stats


# --------------------------------------------------------
# Main Streamlit App
# --------------------------------------------------------
def main():
    # Set the app's page config with a custom title and wide layout
    st.set_page_config(
        page_title="🦅🦅🦅🦅 News from the Hill Dashboard",
        layout="wide"
    )

    # Title for the app
    st.title("🦅🦅🦅🦅 News from the Hill — Congressional Analysis")

    # Load data for Senate and House
    senate_data = load_data("senate_analysis_output")
    house_data = load_data("house_analysis_output")

    # Sidebar for chamber selection
    st.sidebar.title("Chamber Selector")
    chamber = st.sidebar.radio("Choose a Chamber:", ["Senate", "House"])

    if chamber == "Senate":
        data = senate_data
    else:
        data = house_data

    if data["user_stats"].empty:
        st.warning(f"No user stats found for {chamber}. Please check your CSV files.")
        return

    # --------------------------------
    # Overall Chamber Statistics
    # --------------------------------
    st.header(f"Overall {chamber} Statistics")
    overall_stats = get_overall_stats(data["user_stats"])
    if overall_stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Messages", overall_stats["Total Messages"])
        c2.metric("Total Words", overall_stats["Total Words"])
        c3.metric("Combined Avg. Msgs/Day (All Users)", round(overall_stats["Combined Avg. Messages/Day (All Users)"], 2))
        c4.metric("Combined Avg. Words/Day (All Users)", round(overall_stats["Combined Avg. Words/Day (All Users)"], 2))
    else:
        st.warning("No overall stats available.")

    # --------------------------------
    # Overall Word Cloud
    # --------------------------------
    st.subheader(f"Overall Word Cloud ({chamber})")

    df_word_freq = data["word_frequencies"]
    if not df_word_freq.empty:
        df_word_freq = df_word_freq.dropna(subset=['word'])
        df_word_freq['word'] = df_word_freq['word'].astype(str)
    else:
        st.info(f"No word frequencies found for {chamber}.")

    if not df_word_freq.empty:
        # Let the user add a comma-separated list of stopwords (unique key!)
        with st.expander("Word Cloud Stopwords"):
            st.write("Remove specific words from the Word Cloud by listing them here (comma-separated).")
            stopwords_input = st.text_input(
                "Stopwords (comma-separated)",
                value="",
                key="overall_stopwords"
            )
            if stopwords_input.strip():
                user_stopwords = [w.strip().lower() for w in stopwords_input.split(",")]
            else:
                user_stopwords = []

        # Build frequency dict and exclude user stopwords
        word_freq_dict = {}
        for w, freq in zip(df_word_freq['word'], df_word_freq['frequency']):
            if w.lower() not in user_stopwords:
                word_freq_dict[w] = freq

        if len(word_freq_dict) == 0:
            st.info("After removing chosen words, no words remain to display.")
        else:
            wc = generate_wordcloud_from_frequencies(word_freq_dict)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

    # --------------------------------
    # Multi-Word Frequency Search
    # --------------------------------
    st.subheader("Word Frequency Lookup (Across All Users)")
    if not df_word_freq.empty:
        all_unique_words = sorted(df_word_freq['word'].unique())
        selected_words = st.multiselect("Select word(s) to see their frequency:", options=all_unique_words, default=[])
        if selected_words:
            filtered = df_word_freq[df_word_freq['word'].isin(selected_words)]
            if not filtered.empty:
                # Show table with nicer column names
                filtered_display = filtered.rename(columns={
                    'word': 'Word',
                    'frequency': 'Frequency'
                })
                st.dataframe(filtered_display.reset_index(drop=True))

                # Bar chart
                bar_chart = alt.Chart(filtered_display).mark_bar().encode(
                    x=alt.X('Word:O', sort='-y', title='Word'),
                    y=alt.Y('Frequency:Q', title='Frequency'),
                    tooltip=['Word', 'Frequency']
                ).properties(width=600, height=300)
                st.altair_chart(bar_chart, use_container_width=True)
            else:
                st.info("No matching words found.")
        else:
            st.info("Select one or more words above to see frequency details.")

    # --------------------------------
    # Words per Day (with optional rolling avg)
    # --------------------------------
    if not data["words_per_day"].empty and "date" in data["words_per_day"].columns:
        st.subheader("Words per Day (with optional rolling average)")
        st.write("Shows total words used each day (across all users).")

        wpd = data["words_per_day"].copy()
        wpd.rename(columns={"date": "Date", "words": "Words"}, inplace=True)
        wpd = wpd.dropna(subset=["Date"])
        wpd = wpd[wpd["Date"].notnull()].sort_values("Date")

        # Rolling window size slider (unique key)
        with st.expander("Smoothing / Rolling Average (Words)"):
            st.write("Use a rolling window to smooth the daily word counts. Example: a window of 7 shows a 7-day rolling average.")
            words_window_size = st.slider("Rolling window (days) for Words", 
                                          min_value=1, max_value=30, value=7, 
                                          key="rolling_words_window")

        # Add rolling average column
        wpd["RollingAvg_Words"] = wpd["Words"].rolling(window=words_window_size, min_periods=1).mean()

        # Layer chart: raw words + rolling avg
        raw_line = alt.Chart(wpd).mark_line(color="steelblue").encode(
            x=alt.X('Date:T', title="Date"),
            y=alt.Y('Words:Q', title="Total Words"),
            tooltip=["Date:T", "Words:Q"]
        )
        rolling_line = alt.Chart(wpd).mark_line(color="orange").encode(
            x=alt.X('Date:T'),
            y=alt.Y('RollingAvg_Words:Q', title=f"{words_window_size}-Day Rolling Avg (Words)"),
            tooltip=["Date:T", "RollingAvg_Words:Q"]
        )

        layered_words_chart = alt.layer(raw_line, rolling_line).resolve_scale(y='shared').properties(
            width=800,
            height=400
        ).interactive()

        st.altair_chart(layered_words_chart, use_container_width=True)
        st.write("**Blue**: raw daily words, **Orange**: rolling average.")

    # --------------------------------
    # Messages per Day (with optional rolling avg)
    # --------------------------------
    if not data["messages_per_day"].empty and "date" in data["messages_per_day"].columns:
        st.subheader("Messages per Day (with optional rolling average)")

        mpd = data["messages_per_day"].copy()
        mpd.rename(columns={"date": "Date", "messages": "Messages"}, inplace=True)
        mpd = mpd.dropna(subset=["Date"])
        mpd = mpd[mpd["Date"].notnull()].sort_values("Date")

        with st.expander("Smoothing / Rolling Average (Messages)"):
            st.write("Use a rolling window to smooth daily message counts.")
            messages_window_size = st.slider("Rolling window (days) for Messages", 
                                             min_value=1, max_value=30, value=7, 
                                             key="rolling_messages_window")

        mpd["RollingAvg_Msgs"] = mpd["Messages"].rolling(window=messages_window_size, min_periods=1).mean()

        # Create an Altair layered chart: raw + rolling avg
        raw_line = alt.Chart(mpd).mark_line(color="steelblue").encode(
            x=alt.X('Date:T', title="Date"),
            y=alt.Y('Messages:Q', title="Messages"),
            tooltip=["Date:T", "Messages:Q"]
        )

        rolling_line = alt.Chart(mpd).mark_line(color="orange").encode(
            x=alt.X('Date:T', title="Date"),
            y=alt.Y('RollingAvg_Msgs:Q', title=f"{messages_window_size}-Day Rolling Avg (Msgs)"),
            tooltip=["Date:T", "RollingAvg_Msgs:Q"]
        )

        layered_chart = alt.layer(raw_line, rolling_line).resolve_scale(y='shared').properties(
            width=800,
            height=400
        ).interactive()

        st.altair_chart(layered_chart, use_container_width=True)
        st.write("**Blue**: raw daily messages, **Orange**: rolling average.")

    # --------------------------------
    # User-Level Data
    # --------------------------------
    st.header(f"{chamber} User Data")

    user_df = data["user_stats"].copy()
    if not user_df.empty:
        # Rename columns for a nicer UI
        user_df.rename(columns={
            "author": "Author",
            "total_messages": "Total Messages",
            "total_words": "Total Words",
            "avg_messages_per_day": "Avg. Messages/Day",
            "avg_words_per_day": "Avg. Words/Day"
        }, inplace=True)

        user_list = sorted(user_df["Author"].unique().tolist())
        selected_user = st.selectbox("Select a user to see individual stats:", ["(None)"] + user_list)

        if selected_user != "(None)":
            user_row = user_df[user_df["Author"] == selected_user]
            if not user_row.empty:
                st.subheader(f"Stats for {selected_user}")
                st.dataframe(user_row.reset_index(drop=True))

                # User-level word frequencies
                user_wf = data["user_word_frequencies"].copy()
                if not user_wf.empty:
                    user_wf = user_wf.dropna(subset=['author', 'word'])
                    user_wf['word'] = user_wf['word'].astype(str)
                    user_specific_wf = user_wf[user_wf['author'] == selected_user]

                    if not user_specific_wf.empty:
                        st.subheader(f"Word Cloud for {selected_user}")

                        # Let user remove stopwords for this individual's word cloud
                        with st.expander(f"Stopwords for {selected_user}'s word cloud"):
                            user_stopwords_input = st.text_input(
                                "Stopwords (comma-separated)",
                                value="",
                                key=f"user_{selected_user}_stopwords"
                            )
                            if user_stopwords_input.strip():
                                user_stopwords_ = [w.strip().lower() for w in user_stopwords_input.split(",")]
                            else:
                                user_stopwords_ = []

                        freq_dict_user = {}
                        for w, freq in zip(user_specific_wf['word'], user_specific_wf['frequency']):
                            if w.lower() not in user_stopwords_:
                                freq_dict_user[w] = freq

                        if len(freq_dict_user) == 0:
                            st.info("After removing stopwords, no words remain to display.")
                        else:
                            # Generate user-specific word cloud
                            wc_user = generate_wordcloud_from_frequencies(freq_dict_user)
                            fig_wc_user, ax_wc_user = plt.subplots(figsize=(10,5))
                            ax_wc_user.imshow(wc_user, interpolation='bilinear')
                            ax_wc_user.axis("off")
                            st.pyplot(fig_wc_user)

                        # User-specific word frequency lookup
                        st.subheader(f"Search {selected_user}'s Word Frequency")
                        all_user_words = sorted(user_specific_wf['word'].unique())
                        selected_user_words = st.multiselect(
                            "Pick word(s) to see frequency",
                            options=all_user_words,
                            default=[]
                        )
                        if selected_user_words:
                            filtered_user_words = user_specific_wf[user_specific_wf['word'].isin(selected_user_words)]
                            if not filtered_user_words.empty:
                                # Clean up columns
                                filtered_user_words = filtered_user_words.rename(columns={
                                    'author': 'Author',
                                    'word': 'Word',
                                    'frequency': 'Frequency'
                                })
                                st.dataframe(filtered_user_words.reset_index(drop=True))

                                # Bar chart
                                user_bar_chart = alt.Chart(filtered_user_words).mark_bar().encode(
                                    x=alt.X('Word:O', sort='-y', title='Word'),
                                    y=alt.Y('Frequency:Q', title='Frequency'),
                                    tooltip=['Word', 'Frequency']
                                ).properties(width=600, height=300)
                                st.altair_chart(user_bar_chart, use_container_width=True)
                            else:
                                st.info("No matching words found.")
                        else:
                            st.info(f"Select word(s) above to see frequency details.")
                    else:
                        st.info("No word frequency data found for this user.")
                else:
                    st.info("No user_word_frequencies.csv available.")
            else:
                st.warning("No data found for the selected user.")
    else:
        st.info("No user data available for this chamber.")

    st.write("---")
    st.markdown("**🦅🦅🦅🦅 News from the Hill Dashboard**")


# --------------------------------------------------------
# Run the Streamlit app
# --------------------------------------------------------
if __name__ == "__main__":
    main()
