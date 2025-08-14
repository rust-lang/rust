//@ check-pass

#![crate_type = "lib"]

// Test that encoding the hallucinated `DefId` for the `_` const argument doesn't
// ICE (see #133468). This requires this to be a library crate.

pub fn foo() {
    let s: [u8; 10];
    s = [0; _];
}
