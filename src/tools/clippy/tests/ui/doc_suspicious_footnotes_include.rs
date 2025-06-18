//@ error-in-other-file: footnote
//@ no-rustfix
#![warn(clippy::doc_suspicious_footnotes)]
#![doc=include_str!("doc_suspicious_footnotes_include.txt")]
