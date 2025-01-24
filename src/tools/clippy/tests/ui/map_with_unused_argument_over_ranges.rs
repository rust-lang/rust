#![allow(
    unused,
    clippy::redundant_closure,
    clippy::reversed_empty_ranges,
    clippy::identity_op
)]
#![warn(clippy::map_with_unused_argument_over_ranges)]

fn do_something() -> usize {
    todo!()
}

fn do_something_interesting(x: usize, y: usize) -> usize {
    todo!()
}

macro_rules! gen {
    () => {
        (0..10).map(|_| do_something());
    };
}

fn main() {
    // These should be raised
    (0..10).map(|_| do_something());
    (0..10).map(|_foo| do_something());
    (0..=10).map(|_| do_something());
    (3..10).map(|_| do_something());
    (3..=10).map(|_| do_something());
    (0..10).map(|_| 3);
    (0..10).map(|_| {
        let x = 3;
        x + 2
    });
    (0..10).map(|_| do_something()).collect::<Vec<_>>();
    let upper = 4;
    (0..upper).map(|_| do_something());
    let upper_fn = || 4;
    (0..upper_fn()).map(|_| do_something());
    (0..=upper_fn()).map(|_| do_something());
    (2..upper_fn()).map(|_| do_something());
    (2..=upper_fn()).map(|_| do_something());
    (-3..9).map(|_| do_something());
    (9..3).map(|_| do_something());
    (9..=9).map(|_| do_something());
    (1..=1 << 4).map(|_| do_something());
    // These should not be raised
    gen!();
    let lower = 2;
    let lower_fn = || 2;
    (lower..upper_fn()).map(|_| do_something()); // Ranges not starting at zero not yet handled
    (lower_fn()..upper_fn()).map(|_| do_something()); // Ranges not starting at zero not yet handled
    (lower_fn()..=upper_fn()).map(|_| do_something()); // Ranges not starting at zero not yet handled
    (0..10).map(|x| do_something_interesting(x, 4)); // Actual map over range
    "Foobar".chars().map(|_| do_something()); // Not a map over range
    // i128::MAX == 340282366920938463463374607431768211455
    (0..=340282366920938463463374607431768211455).map(|_: u128| do_something()); // Can't be replaced due to overflow
}

#[clippy::msrv = "1.27"]
fn msrv_1_27() {
    (0..10).map(|_| do_something());
}

#[clippy::msrv = "1.28"]
fn msrv_1_28() {
    (0..10).map(|_| do_something());
}

#[clippy::msrv = "1.81"]
fn msrv_1_82() {
    (0..10).map(|_| 3);
}
