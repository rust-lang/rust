#![allow(dead_code)]

use std::pin::Pin;

struct Foo;

impl Foo {
    fn foo(self: Pin<&mut Self>) {
    }
    fn foo_sugar(&pin mut self) {} //~ ERROR pinned reference syntax is experimental
    fn foo_sugar_const(&pin const self) {} //~ ERROR pinned reference syntax is experimental
}

fn foo(mut x: Pin<&mut Foo>) {
    Foo::foo_sugar(x.as_mut());
    Foo::foo_sugar_const(x.as_ref());
    let _y: &pin mut Foo = x; //~ ERROR pinned reference syntax is experimental
}

fn foo_sugar(_: &pin mut Foo) {} //~ ERROR pinned reference syntax is experimental

fn bar(x: Pin<&mut Foo>) {
    foo(x);
    foo(x); //~ ERROR use of moved value: `x`
}

fn baz(mut x: Pin<&mut Foo>) {
    x.foo();
    x.foo(); //~ ERROR use of moved value: `x`
}

fn baz_sugar(_: &pin const Foo) {} //~ ERROR pinned reference syntax is experimental

#[cfg(any())]
mod not_compiled {
    use std::pin::Pin;

    struct Foo;

    impl Foo {
        fn foo(self: Pin<&mut Self>) {
        }
        fn foo_sugar(&pin mut self) {} //~ ERROR pinned reference syntax is experimental
        fn foo_sugar_const(&pin const self) {} //~ ERROR pinned reference syntax is experimental
    }

    fn foo(mut x: Pin<&mut Foo>) {
        Foo::foo_sugar(x.as_mut());
        Foo::foo_sugar_const(x.as_ref());
        let _y: &pin mut Foo = x; //~ ERROR pinned reference syntax is experimental
    }

    fn foo_sugar(_: &pin mut Foo) {} //~ ERROR pinned reference syntax is experimental

    fn bar(x: Pin<&mut Foo>) {
        foo(x);
        foo(x);
    }

    fn baz(mut x: Pin<&mut Foo>) {
        x.foo();
        x.foo();
    }

    fn baz_sugar(_: &pin const Foo) {} //~ ERROR pinned reference syntax is experimental
}

fn main() {}
