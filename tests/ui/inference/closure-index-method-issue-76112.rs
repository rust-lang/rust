//! Regression test for https://github.com/rust-lang/rust/issues/76112.
//! A closure argument used as an index should be inferred from a later call to the closure.

//@ check-pass

use std::ops::Add;

struct Lhs;

impl Add<i32> for Lhs {
    type Output = i64;

    fn add(self, rhs: i32) -> Self::Output {
        rhs.into()
    }
}

#[derive(Copy, Clone)]
struct Holder;

impl Holder {
    fn take<F>(self, value: F) -> F {
        value
    }
}

fn expect_usize_to_i64<F: Fn(usize) -> i64>(_: F) {}

#[derive(Copy, Clone)]
struct Value;

impl Value {
    fn get(self) -> i32 {
        0
    }
}

fn main() {
    let array: [i64; 1] = [0];
    let get = |index| array[index].pow(1);

    let value: i64 = get(0);
    assert_eq!(value, 0);

    let options = [Some(1i32)];
    let map = |index| options[index].map(|value| value + 1);

    let value: Option<i32> = map(0);
    assert_eq!(value, Some(2));

    let get_chained = |index| array[index].wrapping_add(1).pow(1);
    let value: i64 = get_chained(0);
    assert_eq!(value, 1);

    let add = |rhs| (Lhs + rhs).pow(1);
    let value: i64 = add(2);
    assert_eq!(value, 2);

    // The inner closure waits until `take` has constrained its argument and the outer call has
    // constrained `make_nested`.
    let holders = [Holder];
    let make_nested = |index| holders[index].take(|inner| array[inner].pow(1));
    expect_usize_to_i64(make_nested(0));

    let direct_receiver = |value| value.get();
    let _: i32 = direct_receiver(Value);

    let referenced_receiver = |value: &_| (*value).get();
    let _: i32 = referenced_receiver(&Value);

    let make_value = || 1u32;
    let _: u32 = make_value().count_ones();

    let mut inferred_from_body = None;
    let _set = || inferred_from_body = Some(1u32);
    let _: u32 = inferred_from_body.unwrap().count_ones();

    let mut tuple_inferred_from_body = None;
    let _set = || tuple_inferred_from_body = Some((1u32,));
    let _: u32 = tuple_inferred_from_body.unwrap().0;

    // A const block has its own body owner, so it must finish any closures deferred within it.
    const {
        let _unused = |value| {
            let _: u32 = value;
            value
        };
    }
}
