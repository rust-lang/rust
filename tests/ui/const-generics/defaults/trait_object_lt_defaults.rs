//@ aux-build:trait_object_lt_defaults_lib.rs
//@ run-pass
#![allow(dead_code)]
extern crate trait_object_lt_defaults_lib;

// Tests that `A<'a, 3, dyn Test>` is short for `A<'a, 3, dyn Test + 'a>`
// and `Foo<'a, 3, dyn Test>` is short for `Foo<'a, 3, dyn Test + 'a>`
// Test is in `const-generics/defaults` because it relies on param ordering

trait Test {}

struct A<'a, const N: usize, T: ?Sized + 'a>(&'a T, [(); N]);
fn blah<'a>(mut a: A<'a, 3, dyn Test>, arg: &'a (dyn Test + 'a)) {
    a.0 = arg;
}

fn other_blah<'a>(
    mut a: trait_object_lt_defaults_lib::Foo<'a, 3, dyn Test>,
    arg: &'a (dyn Test + 'a),
) {
    a.0 = arg;
}

fn main() {}
