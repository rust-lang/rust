#![feature(coverage_attribute)]
//@ edition: 2021

// If a closure body consists entirely of a single bang-macro invocation, the
// body span ends up inside the macro-expansion, so we need to un-expand it
// back to the declaration site.
static NO_BLOCK: fn() = || println!("hello");

static WITH_BLOCK: fn() = || {
    println!("hello");
};

#[coverage(off)]
fn main() {
    NO_BLOCK();
    WITH_BLOCK();
}
