// Test the error message resulting from a cycle in solving `Foo:
// Sized`. The specifics of the message will of course but the main
// thing we want to preserve is that:
//
// 1. the message should appear attached to one of the structs
//    defined in this file;
// 2. it should elaborate the steps that led to the cycle.

struct Baz { q: Option<Foo> }
//~^ ERROR recursive type `Baz` has infinite size
struct Foo { q: Option<Baz> }
//~^ ERROR recursive type `Foo` has infinite size

impl Foo { fn bar(&self) {} }

fn main() {}
