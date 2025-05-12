//@ check-pass

#![feature(auto_traits, negative_impls)]

auto trait Foo {}
auto trait Bar {}

struct NeedsOutlives<'a, T>(&'a T);

impl<'a, T: 'a> !Foo for NeedsOutlives<'a, T> {}

// Leaving out the lifetime bound
impl<'a, T> !Bar for NeedsOutlives<'a, T> {}

struct NeedsSend<T: Send>(T);

impl<T: Send> !Foo for NeedsSend<T> {}

// Leaving off the trait bound
impl<T> !Bar for NeedsSend<T> {}

fn main() {}
