#![feature(custom_inner_attributes)]
#![rustfmt::skip]
#![warn(clippy::doc_paragraphs_missing_punctuation)]
//@no-rustfix

enum UnfixableTrailers {
    /// Sometimes the doc comment ends with parentheses (like this)
    //~^ doc_paragraphs_missing_punctuation
    EndsWithParens,
    /// This comment ends with "a quote"
    //~^ doc_paragraphs_missing_punctuation
    QuoteFailing,
}
