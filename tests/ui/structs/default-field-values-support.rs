//@ run-pass
#![feature(default_field_values, generic_const_exprs)]
#![allow(unused_variables, dead_code, incomplete_features)]

pub struct S;

#[derive(Default)]
pub struct Foo {
    pub bar: S = S,
    pub baz: i32 = 42 + 3,
}

#[derive(Default)]
pub enum Bar {
    #[default]
    Foo {
        bar: S = S,
        baz: i32 = 42 + 3,
    }
}

#[derive(Default)]
pub struct Qux<A, const C: i32> {
    bar: S = Qux::<A, C>::S,
    baz: i32 = foo(),
    bat: i32 = <Qux<A, C> as T>::K,
    baq: i32 = Self::K,
    bay: i32 = C,
    bak: Vec<A> = Vec::new(),
}

impl<A, const C: i32> Qux<A, C> {
    const S: S = S;
}

trait T {
    const K: i32;
}

impl<A, const C: i32> T for Qux<A, C> {
    const K: i32 = 2;
}

const fn foo() -> i32 {
    42
}

fn main () {
    let x = Foo { .. };
    let y = Foo::default();
    let z = Foo { baz: 1, .. };

    assert_eq!(45, x.baz);
    assert_eq!(45, y.baz);
    assert_eq!(1, z.baz);

    let x = Bar::Foo { .. };
    let y = Bar::default();
    let z = Bar::Foo { baz: 1, .. };

    assert!(matches!(Bar::Foo { bar: S, baz: 45 }, x));
    assert!(matches!(Bar::Foo { bar: S, baz: 45 }, y));
    assert!(matches!(Bar::Foo { bar: S, baz: 1 }, z));

    let x = Qux::<i32, 4> { .. };
    assert!(matches!(Qux::<i32, 4> { bar: S, baz: 42, bat: 2, baq: 2, bay: 4, .. }, x));
    assert!(x.bak.is_empty());
}
