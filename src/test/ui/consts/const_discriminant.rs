// run-pass
#![feature(const_discriminant)]

use std::mem::{discriminant, Discriminant};

enum Test {
    A(u8),
    B,
    C { a: u8, b: u8 },
}

const TEST_A: Discriminant<Test> = discriminant(&Test::A(5));
const TEST_A_OTHER: Discriminant<Test> = discriminant(&Test::A(17));
const TEST_B: Discriminant<Test> = discriminant(&Test::B);

fn main() {
    assert_eq!(TEST_A, TEST_A_OTHER);
    assert_eq!(TEST_A, discriminant(&Test::A(17)));
    assert_eq!(TEST_B, discriminant(&Test::B));
    assert_ne!(TEST_A, TEST_B);
    assert_ne!(TEST_B, discriminant(&Test::C { a: 42, b: 7 }));
}
