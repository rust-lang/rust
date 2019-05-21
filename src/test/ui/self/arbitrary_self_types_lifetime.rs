// compile-pass

#![feature(arbitrary_self_types)]

use std::pin::Pin;

#[derive(Debug)]
struct Foo;
#[derive(Debug)]
struct Bar<'a>(&'a Foo);

impl std::ops::Deref for Bar<'_> {
    type Target = Foo;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Foo {
    fn a(&self) -> Bar<'_> {
        Bar(self)
    }

    fn b(c: &Self) -> Bar<'_> {
        Bar(c)
    }

    fn c(self: Bar<'_>) -> Bar<'_> {
        self
    }

    fn d(e: Bar<'_>) -> Bar<'_> {
        e
    }

    fn e(self: &Self) -> Bar<'_> {
        Bar(self)
    }

    fn f(self: Bar<'_>) -> impl std::fmt::Debug + '_ {
        self
    }
}

impl<'a> Bar<'a> {
    fn a(self: Bar<'a>, f: &Foo) -> (Bar<'a>, &Foo) { (self, f) }
    fn b(self: Self, f: &Foo) -> (Bar<'a>, &Foo) { (self, f) }
    fn d(self: Bar<'a>, f: &Foo) -> (Self, &Foo) { (self, f) }
}

impl Bar<'_> {
    fn e(self: Self, f: &Foo) -> (Self, &Foo) { (self, f) }
}

struct Baz<T: Unpin> {
    field: T,
}

impl<T: Unpin> Baz<T> {
    fn field(self: Pin<&mut Self>) -> Pin<&mut T> {
        let this = Pin::get_mut(self);
        Pin::new(&mut this.field)
    }
}

fn main() {
    let foo = Foo;
    { foo.a() };
    { Foo::b(&foo) };
    { Bar(&foo).c() };
    { Foo::d(Bar(&foo)) };
    { foo.e() };
    { Bar(&foo).f() };
    let mut baz = Baz { field: 0u8 };
    { Pin::new(&mut baz).field() };
}
