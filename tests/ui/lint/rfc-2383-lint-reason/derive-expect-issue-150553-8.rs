// The `#[expect]` sharing with derive-generated code only applies to builtin derive
// macros, recognized through the def collector's record of the expansion that created
// the impl. A proc-macro derive cannot opt into it, no matter which spans it assigns
// to its output tokens: the expectations on the derive input stay unfulfilled.

//@ check-pass
//@ proc-macro: derive_with_spans.rs

extern crate derive_with_spans;
use derive_with_spans::{WithCallSite, WithDefSite, WithMixedSite};

trait Trait {
    fn method(&self);
}

#[expect(unused_variables)] //~ WARN this lint expectation is unfulfilled
#[derive(WithCallSite)]
struct A;

#[expect(unused_variables)] //~ WARN this lint expectation is unfulfilled
#[derive(WithMixedSite)]
struct B;

#[expect(unused_variables)] //~ WARN this lint expectation is unfulfilled
#[derive(WithDefSite)]
struct C;

fn main() {
    A.method();
    B.method();
    C.method();
}
