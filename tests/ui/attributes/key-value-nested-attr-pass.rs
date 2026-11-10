// Test that we allow active attributes[^1] on the "value" of key-value attributes, that we expand
// expand them and that it's the expansion that ends up getting used as the a final value.
//
// [^1]: Barring `cfg` attributes which would be too footgun-y.

//@ proc-macro: replace.rs
//@ proc-macro: derive_macro_with_helper.rs
//@ edition: 2018..
//@ check-pass

#![feature(stmt_expr_attributes, proc_macro_hygiene)]

#![doc = #[replace::identity] "original"] // OK

#![doc = #[replace::with("replacement")] "original"] // OK

// Indeed, the RHS isn't an unsuffixed literal (without any inert attributes) but a more complex
// expression. That's perfectly fine because it's the expansion that counts (here: `"literal"`).
#![doc = #[replace::with("literal")] (a + b * c)] // OK

#![doc = #[replace::with("C")] #[replace::with("B")] "A"] // OK

#![doc =
    #[replace::with(
        #[replace::with("C")]
        "B"
    )]
    "A"
] // OK

// While `#[repr(C)]` and `#[allow(dead_code)]` are *inert* attributes that are usually forbidden in
// this context (see `key-value-nested-attr-fail.rs`), macro attribute `replace::with` replaces the
// entire expression including attributes with literal `"text"`.
#![doc =
    #[repr(C)]
    #[replace::with("replacement")]
    #[allow(dead_code)]
    "original"
] // OK

// While inert attributes `repr` & `allow` are malformed, macro attribute `replace::with` replaces
// the entire expression including attributes with literal `"text"` before they get semantically
// validated. Whether this is considered a feature or not is an open question at this time of
// writing. CC <https://github.com/rust-lang/rust/issues/165073>.
#![doc =
    #[repr] // (malformed)
    #[allow = "this"] // (malformed)
    #[replace::with("replacement")]
    "original"
] // OK

// The expansion of `cfg_attr` leaves behind *traces* in the form of "fake" attrs. We ought not get
// confused by them and accidentally reject them. After all, we want to allow active attributes here
// (barring `cfg`).
#![doc = #[cfg_attr(false, allow(dead_code))] "literal"] // OK
#![doc = #[cfg_attr(true, replace::with("literal"))] ()] // OK

// Tool attribute.
#[clippy::immaterial = #[replace::with("replacement")] "original"] // OK
mod target {}

// Helper attribute.
#[derive(derive_macro_with_helper::Derive)]
#[arg =
    #[repr(C)]
    #[allow(dead_code)]
    #[replace::with("replacement")]
    "original"
] // OK
struct Owner;

fn main() {}
