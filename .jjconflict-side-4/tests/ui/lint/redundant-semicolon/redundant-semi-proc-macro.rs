//@ proc-macro: redundant-semi-proc-macro-def.rs

#![deny(redundant_semicolons)]
extern crate redundant_semi_proc_macro;
use redundant_semi_proc_macro::should_preserve_spans;

#[should_preserve_spans]
fn span_preservation()  {
    let tst = 123;; //~ ERROR unnecessary trailing semicolon
    match tst {
        // Redundant semicolons are parsed as empty tuple exprs
        // for the lint, so ensure the lint doesn't affect
        // empty tuple exprs explicitly in source.
        123 => (),
        _ => ()
    };;; //~ ERROR unnecessary trailing semicolons
}

fn main() {}
