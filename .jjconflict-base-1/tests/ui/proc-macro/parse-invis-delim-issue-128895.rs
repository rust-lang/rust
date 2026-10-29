//@ proc-macro: parse-invis-delim-issue-128895.rs
//@ check-pass

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate parse_invis_delim_issue_128895;

trait Comparable {}

parse_invis_delim_issue_128895::main!();

fn main() {}
