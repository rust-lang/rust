//@ edition: 2021
//@ run-rustfix
//@ rustfix-only-machine-applicable
//@ aux-build:migration_lint_macros.rs
#![feature(mut_ref)]
#![allow(incomplete_features, unused)]
#![deny(rust_2024_incompatible_pat)]

extern crate migration_lint_macros;

struct Foo<T>(T);

// Tests type equality in a way that avoids coercing `&&T` to `&T`.
trait Eq<T> {}
impl<T> Eq<T> for T {}
fn assert_type_eq<T, U: Eq<T>>(_: T, _: U) {}

fn main() {
    let Foo(x) = &Foo(0);
    assert_type_eq(x, &0u8);

    let Foo(x) = &mut Foo(0);
    assert_type_eq(x, &mut 0u8);

    let Foo(mut x) = &Foo(0);
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(x, 0u8);

    let Foo(mut x) = &mut Foo(0);
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(x, 0u8);

    let Foo(ref x) = &Foo(0);
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(x, &0u8);

    let Foo(ref x) = &mut Foo(0);
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(x, &0u8);

    let &Foo(x) = &Foo(0);
    assert_type_eq(x, 0u8);

    let &mut Foo(x) = &mut Foo(0);
    assert_type_eq(x, 0u8);

    let &Foo(x) = &Foo(&0);
    assert_type_eq(x, &0u8);

    let &mut Foo(x) = &mut Foo(&0);
    assert_type_eq(x, &0u8);

    let Foo(&x) = &Foo(&0);
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(x, 0u8);

    let Foo(&mut x) = &Foo(&mut 0);
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(x, 0u8);

    let Foo(&x) = &mut Foo(&0);
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(x, 0u8);

    let Foo(&mut x) = &mut Foo(&mut 0);
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(x, 0u8);

    if let Some(x) = &&&&&Some(&0u8) {
        assert_type_eq(x, &&0u8);
    }

    if let Some(&x) = &&&&&Some(&0u8) {
        //~^ ERROR: patterns are not allowed to reset the default binding mode
        //~| WARN: this changes meaning in Rust 2024
        assert_type_eq(x, 0u8);
    }

    if let Some(&mut x) = &&&&&Some(&mut 0u8) {
        //~^ ERROR: patterns are not allowed to reset the default binding mode
        //~| WARN: this changes meaning in Rust 2024
        assert_type_eq(x, 0u8);
    }

    if let Some(&x) = &&&&&mut Some(&0u8) {
        //~^ ERROR: patterns are not allowed to reset the default binding mode
        //~| WARN: this changes meaning in Rust 2024
        assert_type_eq(x, 0u8);
    }

    if let Some(&mut Some(Some(x))) = &mut Some(&mut Some(&mut Some(0u8))) {
        //~^ ERROR: patterns are not allowed to reset the default binding mode
        //~| WARN: this changes meaning in Rust 2024
        assert_type_eq(x, &mut 0u8);
    }

    struct Struct<A, B, C> {
        a: A,
        b: B,
        c: C,
    }

    let Struct { a, mut b, c } = &Struct { a: 0, b: 0, c: 0 };
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(a, &0u32);
    assert_type_eq(b, 0u32);

    let Struct { a: &a, b, ref c } = &Struct { a: &0, b: &0, c: &0 };
    //~^ ERROR: patterns are not allowed to reset the default binding mode
    //~| WARN: this changes meaning in Rust 2024
    assert_type_eq(a, 0u32);
    assert_type_eq(b, &&0u32);
    assert_type_eq(c, &&0u32);

    if let Struct { a: &Some(a), b: Some(&b), c: Some(c) } =
        //~^ ERROR: patterns are not allowed to reset the default binding mode
        //~| WARN: this changes meaning in Rust 2024
        &(Struct { a: &Some(&0), b: &Some(&0), c: &Some(&0) })
    {
        assert_type_eq(a, &0u32);
        assert_type_eq(b, 0u32);
        assert_type_eq(c, &&0u32);
    }

    match &(Some(0), Some(0)) {
        // The two patterns are the same syntactically, but because they're defined in different
        // editions they don't mean the same thing.
        (Some(mut x), migration_lint_macros::mixed_edition_pat!(y)) => {
            //~^ ERROR: patterns are not allowed to reset the default binding mode
            assert_type_eq(x, 0u32);
            assert_type_eq(y, 0u32);
        }
        _ => {}
    }
}
