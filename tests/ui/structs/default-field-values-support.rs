// Exercise the `default_field_values` feature to confirm it interacts correctly with other nightly
// features. In particular, we want to verify that interaction with consts coming from different
// contexts are usable as a default field value.
//@ run-pass
//@ aux-build:struct_field_default.rs
#![feature(const_trait_impl, default_field_values, generic_const_exprs)]
#![allow(unused_variables, dead_code, incomplete_features)]

extern crate struct_field_default as xc;

pub struct S;

// Basic expressions and `Default` expansion
#[derive(Default)]
pub struct Foo {
    pub bar: S = S,
    pub baz: i32 = 42 + 3,
}

// Enum support for deriving `Default` when all fields have default values
#[derive(Default)]
pub enum Bar {
    #[default]
    Foo {
        bar: S = S,
        baz: i32 = 42 + 3,
    }
}

#[const_trait] pub trait ConstDefault {
    fn value() -> Self;
}

impl const ConstDefault for i32 {
    fn value() -> i32 {
        101
    }
}

pub struct Qux<A, const C: i32, X: const ConstDefault> {
    bar: S = Qux::<A, C, X>::S, // Associated constant from inherent impl
    baz: i32 = foo(), // Constant function
    bat: i32 = <Qux<A, C, X> as T>::K, // Associated constant from explicit trait
    baq: i32 = Self::K, // Associated constant from implicit trait
    bay: i32 = C, // `const` parameter
    bak: Vec<A> = Vec::new(), // Associated constant function
    ban: X = X::value(), // Associated constant function from `const` trait parameter
}

impl<A, const C: i32, X: const ConstDefault> Qux<A, C, X> {
    const S: S = S;
}

trait T {
    const K: i32;
}

impl<A, const C: i32, X: const ConstDefault> T for Qux<A, C, X> {
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

    let x = Qux::<i32, 4, i32> { .. };
    assert!(matches!(
        Qux::<i32, 4, i32> {
            bar: S,
            baz: 42,
            bat: 2,
            baq: 2,
            bay: 4,
            ban: 101,
            ..
        },
        x,
    ));
    assert!(x.bak.is_empty());

    let x = xc::A { .. };
    assert!(matches!(xc::A { a: 42 }, x));
}
