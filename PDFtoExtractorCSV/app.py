import streamlit as st
from helpers import *


def main():
    st.set_page_config(page_title="PDF to Extractor CSV", layout="wide")
    st.title("PDF Extractor to CSV")

    pdf_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    extract_button = st.button("Extract Data")

    if extract_button:
        with st.spinner("Extracting data..."):
            df = create_docs(pdf_files)
            st.write(df)
            # df["AMOUNT"] = df["Net Amount"] + df["Vat Amount"]
            # st.write("Total Amount: ", df["AMOUNT"].sum())
            # st.write("Total Net Amount: ", df["Net Amount"].sum())

            # convert to csv
            convert_to_csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV",
                convert_to_csv,
                "CSV_file.csv",
                "text/csv",
                key="download-csv",
            )

        st.success("Data extracted successfully!")

if __name__ == "__main__":
    main()
