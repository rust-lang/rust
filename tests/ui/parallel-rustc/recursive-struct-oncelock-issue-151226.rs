// Test for #151226, Unable to verify registry association
//
//@ compile-flags: -Z threads=2
//@ compare-output-by-lines
#![allow(todo_macro_calls)]

struct A<T>(std::sync::OnceLock<Self>);
//~^ ERROR recursive type `A` has infinite size
static B: A<()> = todo!();
fn main() {}
