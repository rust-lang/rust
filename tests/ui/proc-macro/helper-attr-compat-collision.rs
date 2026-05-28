//@ proc-macro: test-macros.rs
//@ proc-macro: extra-empty-derive.rs
//@ check-pass

#[macro_use(Empty)]
extern crate test_macros;
#[macro_use(Empty2)]
extern crate extra_empty_derive;

// Testing the behavior of derive attributes with helpers that share the same name.
//
// Normally if the first derive below were absent the call to #[empty_helper] before it it
// introduced by its own derive would produce a future incompat error.
//
// With the extra derive also introducing that attribute in advanced the warning gets supressed.
// Demonstrates a lack of identity to helper attributes, the compiler does not track which derive
// introduced a helper, just that a derive introduced the helper.
#[derive(Empty)]
#[empty_helper]
#[derive(Empty2)]
struct S;

fn main() {}
