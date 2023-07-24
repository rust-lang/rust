// run-pass
#![feature(const_discriminant)]
#![allow(dead_code)]

use std::mem::{discriminant, Discriminant};
use std::hint::black_box;

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
    assert_eq!(TEST_A, discriminant(black_box(&Test::A(17))));
    assert_eq!(TEST_B, discriminant(black_box(&Test::B)));
    assert_ne!(TEST_A, TEST_B);
    assert_ne!(TEST_B, discriminant(black_box(&Test::C { a: 42, b: 7 })));

    assert_eq!(TEST_V, discriminant(black_box(&SingleVariant::V)));
}
