"""
make_docs.py
Generate synthetic bank documents for the RAG part of the agent.
Like make_data.py: fictional content, written around the dataset's columns
(Merchant_Category, Country, Account_Type, Amount), so questions can combine
documents and data.

The documents contain rules and targets, never facts about the data.
Actual numbers always have to come from the dataset tools.

Usage:
  pip install reportlab
  python make_docs.py        # writes PDFs into documents/
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path("documents")
BANK = "Northbridge Bank"

# file name -> list of pages; each page is a list of (style, text)
DOCS = {
    "card_spending_policy.pdf": [
        [
            ("Title", f"{BANK} - Card Spending Policy"),
            ("Normal", "Document CSP-04, version 2.1. Owner: Retail Risk. Effective 1 January 2025."),
            ("Heading2", "1. Purpose"),
            ("Normal", "This policy defines when card transactions must be reviewed by an analyst. "
                       "It applies to all Savings and Current accounts."),
            ("Heading2", "2. Single-transaction review thresholds"),
            ("Normal", "A single transaction must be reviewed when its amount exceeds the threshold "
                       "for its merchant category: Travel 500 CAD, Dining 400 CAD, Grocery 600 CAD, "
                       "Utilities 600 CAD."),
            ("Normal", "Thresholds apply to the transaction amount only. Debit and Credit transactions "
                       "are treated the same way."),
        ],
        [
            ("Heading2", "3. Cross-border transactions"),
            ("Normal", "A transaction is cross-border when its country is not CA. Cross-border "
                       "transactions above 300 CAD require secondary verification, in addition to "
                       "any category threshold in section 2."),
            ("Normal", "The bank expects transactions in FR to stay below 12 percent of all "
                       "transactions. A higher share must be reported to Retail Risk."),
            ("Heading2", "4. Statistical anomaly screening"),
            ("Normal", "Amounts are screened with the interquartile range (IQR) method over all "
                       "transactions: a transaction is an anomaly when its amount is above "
                       "Q3 + 1.5 x IQR or below Q1 - 1.5 x IQR."),
            ("Normal", "If more than 7 percent of transactions are anomalies, the case must be "
                       "escalated to the Risk Committee. An anomaly is not proof of fraud."),
        ],
    ],
    "fraud_review_procedure.pdf": [
        [
            ("Title", f"{BANK} - Fraud Review Procedure"),
            ("Normal", "Document FRP-02. Owner: Fraud Operations. Applies to transactions flagged "
                       "under the Card Spending Policy (CSP-04)."),
            ("Heading2", "1. Timelines"),
            ("Normal", "Every flagged transaction must be reviewed within 48 hours. Transactions "
                       "above 1,500 CAD must be reviewed within 24 hours."),
            ("Heading2", "2. Review steps"),
            ("Normal", "Step 1: confirm the amount, date and merchant category against the source "
                       "record. Step 2: check the customer's transactions in the previous 30 days. "
                       "Step 3: if three or more transactions of the same customer are flagged in "
                       "one month, escalate to a senior analyst."),
        ],
        [
            ("Heading2", "3. Customer contact"),
            ("Normal", "Customers may only be contacted through channels registered on their "
                       "account. Analysts must not ask for passwords or full card numbers."),
            ("Heading2", "4. Data handling"),
            ("Normal", "Customer identifiers, amounts and transaction details must never be entered "
                       "into external websites, search engines or third-party tools. Analysis is "
                       "done only on approved internal systems."),
            ("Normal", "Review notes are kept for seven years for audit purposes."),
        ],
    ],
    "spending_report_2025.pdf": [
        [
            ("Title", f"{BANK} - 2025 Spending Targets and Outlook"),
            ("Normal", "Prepared by Retail Analytics for the Executive Committee, January 2025."),
            ("Heading2", "1. Portfolio targets for 2025"),
            ("Normal", "Travel should represent less than 25 percent of total card spending by "
                       "amount. No single merchant category should exceed 30 percent of total "
                       "spending."),
            ("Normal", "The average transaction amount for Savings accounts should stay below 110 CAD."),
            ("Heading2", "2. Outlook"),
            ("Normal", "Management expects Travel spending to peak in the summer months, June to "
                       "August, following the usual seasonal pattern. Utilities spending is expected "
                       "to be highest in winter."),
        ],
        [
            ("Heading2", "3. Monitoring"),
            ("Normal", "Targets are checked quarterly against the transaction dataset. Any target "
                       "that is missed must be explained in the next quarterly report, with the "
                       "category, the actual figure and the target."),
            ("Normal", "External factors such as interest rates and inflation should be noted when "
                       "they help explain a missed target."),
        ],
    ],
}


def build(name: str, pages: list) -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, page in enumerate(pages):
        if i > 0:
            story.append(PageBreak())       # one section group per page, so page numbers mean something
        for style, text in page:
            story.append(Paragraph(text, styles[style]))
            story.append(Spacer(1, 8))
    SimpleDocTemplate(str(OUT / name), pagesize=letter, title=name).build(story)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, pages in DOCS.items():
        build(name, pages)
        print(f"wrote {OUT / name} ({len(pages)} pages)")