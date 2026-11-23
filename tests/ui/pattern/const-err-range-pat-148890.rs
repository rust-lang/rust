//! Regression test for <https://github.com/rust-lang/rust/issues/148890>

impl std::ops::Neg for u128 {}
//~^ ERROR only traits defined in the current crate can be implemented for primitive types

fn foo(-128..=127: u128) {}

fn bar(x: u128) {
    match x {
        -128 => {}
        -127..127 => {}
        _ => {}
    }
}

fn main() {}
