#![allow(dead_code)]

// Test that we report an error for unused type parameters in types and traits,
// and that we offer a helpful suggestion.

struct SomeStruct<A> {
    //~^ ERROR parameter `A` is never used
    x: u32,
}

enum SomeEnum<A> {
    //~^ ERROR parameter `A` is never used
    Nothing,
}

// Here T might *appear* used, but in fact it isn't.
enum ListCell<T> {
    Cons(Box<ListCell<T>>),
    //~^ ERROR parameter `T` is only used recursively
    Nil,
}

// Example of grounded use of T, which should not trigger an error.
enum ListCellGrounded<T> {
    Cons(Box<ListCellGrounded<T>>),
    Value(T),
    Nil,
}

struct RecursiveInvariant<T>(*mut RecursiveInvariant<T>);
//~^ ERROR parameter `T` is only used recursively

struct SelfTyAlias<T>(Box<Self>);
//~^ ERROR parameter `T` is only used recursively

struct WithBounds<T: Sized> {}
//~^ ERROR parameter `T` is never used

struct WithWhereBounds<T>
//~^ ERROR parameter `T` is never used
where
    T: Sized, {}

struct WithOutlivesBounds<T: 'static> {}
//~^ ERROR parameter `T` is never used

struct DoubleNothing<T> {
    //~^ ERROR parameter `T` is never used
    s: SomeStruct<T>,
}

fn main() {}
