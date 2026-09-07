#![feature(coverage_attribute)]
//@ edition: 2024
//@ min-llvm-version: 23

// Basic test for method calls and chained method calls.

#[rustfmt::skip]
fn call_method() {
    let thing = Thing;

    thing
        .
        method
        (
            "arg"
        )
    ;

    thing
        .
        method
        (
            "arg"
        )
        .
        method
        (
            "arg"
        )
    ;
}

struct Thing;

#[coverage(off)]
impl Thing {
    fn method(&self, _arg: &str) -> &Self {
        self
    }
}

#[coverage(off)]
fn main() {
    call_method();
}
