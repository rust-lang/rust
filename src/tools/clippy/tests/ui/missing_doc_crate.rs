//@ check-pass

#![warn(clippy::missing_docs_in_private_items)]
#![allow(clippy::doc_include_without_cfg)]
#![doc = include_str!("../../README.md")]

fn main() {}
