//@ aux-build:deny-macro.rs
//@ check-pass

// Ensure that when a macro (or normal code) does #[deny] inside a #[forbid]
// context, no error is emitted, as both parties agree on the treatment of the lint.

#![forbid(unsafe_code)]

extern crate deny_macro;

fn main() {
    deny_macro::emit_deny! {}

    #[deny(unsafe_code)]
    let _ = 0;
}
