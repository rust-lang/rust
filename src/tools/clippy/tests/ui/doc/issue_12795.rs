#![warn(clippy::doc_markdown)]

//! A comment with a_b(x) and a_c in it and (a_b((c)) ) too and (maybe a_b((c)))
//~^ doc_markdown
//~| doc_markdown
//~| doc_markdown
//~| doc_markdown

pub fn main() {}
