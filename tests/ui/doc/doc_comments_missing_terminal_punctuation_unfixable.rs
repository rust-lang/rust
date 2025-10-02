#![feature(custom_inner_attributes)]
#![rustfmt::skip]
#![warn(clippy::doc_comments_missing_terminal_punctuation)]
//@no-rustfix

enum UnfixableTrailers {
    /// Sometimes the doc comment ends with parentheses (like this)
    //~^ doc_comments_missing_terminal_punctuation
    EndsWithParens,
    /// This comment ends with "a quote"
    //~^ doc_comments_missing_terminal_punctuation
    QuoteFailing,
}
