//! Test that `impls_reborrow.method()` performs a reborrow.
//@ run-pass

#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct Foo<'a>(PhantomData<&'a ()>);
#[derive(Clone, Copy)]
struct FooRef<'a>(PhantomData<&'a ()>);

impl FooRef<'_> {
    fn eat_foo_ref(self) {}
}

impl Foo<'_> {
    fn eat_foo(self) {}
}

impl<'a> Reborrow for Foo<'a> {}

impl<'a> CoerceShared<FooRef<'a>> for Foo<'a> {}

fn main() {
    let x = Foo(PhantomData);
    let _reborrow = <Foo>::eat_foo(x);
    let _reborrow_with_method = (x.eat_foo(), <FooRef>::eat_foo_ref(x));
}
