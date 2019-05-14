#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete and may cause the compiler to crash

use std::fmt::Debug;

const FOO: impl Debug + Clone + PartialEq<i32> = 42;

static BAR: impl Debug + Clone + PartialEq<i32> = 42;

fn a<T: Clone>(x: T) {
    let y: impl Clone = x;
    let _ = y.clone();
}

fn b<T: Clone>(x: T) {
    let f = move || {
        let y: impl Clone = x;
        let _ = y.clone();
    };
    f();
}

trait Foo<T: Clone> {
    fn a(x: T) {
        let y: impl Clone = x;
        let _ = y.clone();
    }
}

impl<T: Clone> Foo<T> for i32 {
    fn a(x: T) {
        let y: impl Clone = x;
        let _ = y.clone();
    }
}

fn main() {
    let foo: impl Debug + Clone + PartialEq<i32> = 42;

    assert_eq!(FOO.clone(), 42);
    assert_eq!(BAR.clone(), 42);
    assert_eq!(foo.clone(), 42);

    a(42);
    b(42);
    i32::a(42);
}
