// run-pass
#![feature(const_discriminant)]
#![allow(dead_code)]

use std::mem::{discriminant, Discriminant};

// `discriminant(const_expr)` may get const-propagated.
// As we want to check that const-eval is equal to ordinary exection,
// we wrap `const_expr` with a function which is not const to prevent this.
#[inline(never)]
fn identity<T>(x: T) -> T { x }

enum Test {
    A(u8),
    B,
    C { a: u8, b: u8 },
}

const TEST_A: Discriminant<Test> = discriminant(&Test::A(5));
const TEST_A_OTHER: Discriminant<Test> = discriminant(&Test::A(17));
const TEST_B: Discriminant<Test> = discriminant(&Test::B);

enum Void {}

enum SingleVariant {
    V,
    Never(Void),
}

const TEST_V: Discriminant<SingleVariant> = discriminant(&SingleVariant::V);

fn main() {
    assert_eq!(TEST_A, TEST_A_OTHER);
    assert_eq!(TEST_A, discriminant(identity(&Test::A(17))));
    assert_eq!(TEST_B, discriminant(identity(&Test::B)));
    assert_ne!(TEST_A, TEST_B);
    assert_ne!(TEST_B, discriminant(identity(&Test::C { a: 42, b: 7 })));

    assert_eq!(TEST_V, discriminant(identity(&SingleVariant::V)));
}
