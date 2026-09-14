//@ check-pass
//@ edition: 2021
//@ proc-macro: unused-parens-bound-proc-macro.rs

// Regression test for #144378.
//
// A proc-macro can synthesize parentheses around a trait-object bound while
// reusing a span from its input. That span does not actually point at the
// parentheses, so the `unused_parens` lint must not blindly trim its first and
// last byte: doing so produced an invalid suggestion (e.g. turning a field
// `val: u8` into `al: u`) and, when the reused span started or ended on a
// multibyte character, ICEd by slicing through that character.

#![deny(unused_parens)]
#![allow(uncommon_codepoints)]

extern crate unused_parens_bound_proc_macro;

// The generated `&dyn (Send)` reuses the span of the identifier `é`, whose
// first byte is the start of a two-byte character. Before the fix, trimming one
// byte off the front of that span sliced through `é` and ICEd; now the lint is
// skipped because the source span is not wrapped in parentheses.
unused_parens_bound_proc_macro::emit_parenthesized_bound!(é);

fn main() {}
