#![allow(dead_code)]

// Test that we report an error for unused type parameters in types and traits,
// and that we offer a helpful suggestion.

struct SomeStruct<A> { x: u32 }
//~^ ERROR parameter `A` is never used

enum SomeEnum<A> { Nothing }
//~^ ERROR parameter `A` is never used

// Here T might *appear* used, but in fact it isn't.
enum ListCell<T> {
    Cons(Box<ListCell<T>>),
    //~^ ERROR parameter `T` is only used recursively
    Nil
}

struct SelfTyAlias<T>(Box<Self>);
//~^ ERROR parameter `T` is only used recursively

struct WithBounds<T: Sized> {}
//~^ ERROR parameter `T` is never used

struct WithWhereBounds<T> where T: Sized {}
//~^ ERROR parameter `T` is never used

struct WithOutlivesBounds<T: 'static> {}
//~^ ERROR parameter `T` is never used

struct DoubleNothing<T> {
//~^ ERROR parameter `T` is never used
    s: SomeStruct<T>,
}

fn main() {}
