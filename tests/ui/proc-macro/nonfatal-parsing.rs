//@ proc-macro: nonfatal-parsing.rs
//@ needs-unwind
//@ edition: 2024
//@ dont-require-annotations: ERROR
//@ ignore-backends: gcc
//@ revisions: in_macro standalone
//@[in_macro] check-fail
//@[standalone] run-pass
//@[standalone] check-run-results
// FIXME: should be a run-pass test once invalidly parsed tokens no longer result in diagnostics
#![feature(proc_macro_standalone)]

extern crate proc_macro;
extern crate nonfatal_parsing;

#[path = "auxiliary/nonfatal-parsing-body.rs"]
mod body;

fn main() {
    #[cfg(in_macro)]
    nonfatal_parsing::run!();

    #[cfg(standalone)]
    proc_macro::enable_standalone();
    #[cfg(standalone)]
    body::run();
}
