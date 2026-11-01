//@ proc-macro: nonfatal-parsing.rs
//@ needs-unwind
//@ edition: 2024
//@ dont-require-annotations: ERROR
//@ ignore-backends: gcc
// FIXME: should be a run-pass test once invalidly parsed tokens no longer result in diagnostics

extern crate proc_macro;
extern crate nonfatal_parsing;

#[path = "auxiliary/nonfatal-parsing-body.rs"]
mod body;

fn main() {
    nonfatal_parsing::run!();
    // FIXME: enable this once the standalone backend exists
    // https://github.com/rust-lang/rust/issues/130856
    // body::run();
}
