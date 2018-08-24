#![allow(dead_code)]

// Test that we report an error for unused type parameters in types and traits,
// and that we offer a helpful suggestion.

struct SomeStruct<A> { x: u32 }
//~^ ERROR parameter `A` is never used

enum SomeEnum<A> { Nothing }
//~^ ERROR parameter `A` is never used

// Here T might *appear* used, but in fact it isn't.
enum ListCell<T> {
//~^ ERROR parameter `T` is never used
    Cons(Box<ListCell<T>>),
    Nil
}

fn main() {}
