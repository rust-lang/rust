// Test that AsRepr is automatically implemented to convert an int-repr'd enum into its
// discriminant.

// run-pass
// gate-test-enum_as_repr

#![feature(enum_as_repr)]

use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, PartialEq, Eq)]
#[repr(u8)]
enum PositiveNumber {
    Zero,
    One,
}

#[derive(Debug, PartialEq, Eq)]
#[repr(i8)]
enum Number {
    MinusOne = -1,
    Zero,
    One,
    Four = 4,
}

static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, PartialEq, Eq)]
#[repr(usize)]
enum DroppableNumber {
    Zero,
    One,
}

impl Drop for DroppableNumber {
    fn drop(&mut self) {
        DROP_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    use std::enums::AsRepr;

    let n = PositiveNumber::Zero.as_repr();
    assert_eq!(n, 0);
    let n = PositiveNumber::One.as_repr();
    assert_eq!(n, 1);

    let n = std::mem::discriminant(&PositiveNumber::Zero).as_repr();
    assert_eq!(n, 0_u8);
    let n = std::mem::discriminant(&PositiveNumber::One).as_repr();
    assert_eq!(n, 1_u8);

    let n = Number::MinusOne.as_repr();
    assert_eq!(n, -1);
    let n = Number::Zero.as_repr();
    assert_eq!(n, 0);
    let n = Number::One.as_repr();
    assert_eq!(n, 1);
    let n = Number::Four.as_repr();
    assert_eq!(n, 4);

    let n = std::mem::discriminant(&Number::Zero).as_repr();
    assert_eq!(n, 0_i8);
    let n = std::mem::discriminant(&Number::One).as_repr();
    assert_eq!(n, 1_i8);

    assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 0);
    {
        let n = DroppableNumber::Zero;
        assert_eq!(n.as_repr(), 0);
    }
    assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1);
    {
        let n = DroppableNumber::One;
        assert_eq!(n.as_repr(), 1);
    }
    assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 2);
}
