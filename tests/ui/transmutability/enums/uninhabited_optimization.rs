//@ check-pass
//! Tests that we do not regress rust-lang/rust#125811
#![feature(transmutability)]

fn assert_transmutable<T>()
where
    (): std::mem::TransmuteFrom<T>
{}

enum Uninhabited {}

enum SingleInhabited {
    X,
    Y(Uninhabited)
}

enum SingleUninhabited {
    X(Uninhabited),
    Y(Uninhabited),
}

enum MultipleUninhabited {
    X(u8, Uninhabited),
    Y(Uninhabited, u16),
}

fn main() {
    assert_transmutable::<Uninhabited>();
    assert_transmutable::<SingleInhabited>();
    assert_transmutable::<SingleUninhabited>();
    assert_transmutable::<MultipleUninhabited>();
}
