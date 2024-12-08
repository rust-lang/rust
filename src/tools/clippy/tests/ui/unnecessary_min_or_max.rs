//@aux-build:external_consts.rs

#![allow(unused)]
#![warn(clippy::unnecessary_min_or_max)]
#![allow(clippy::identity_op)]

extern crate external_consts;

const X: i32 = 1;

fn main() {
    // Both are Literals
    let _ = (-6_i32).min(9);
    let _ = (-6_i32).max(9);
    let _ = 9_u32.min(6);
    let _ = 9_u32.max(6);
    let _ = 6.min(7_u8);
    let _ = 6.max(7_u8);

    let x: u32 = 42;
    // unsigned with zero
    let _ = 0.min(x);
    let _ = 0.max(x);
    let _ = x.min(0_u32);
    let _ = x.max(0_u32);

    let x: i32 = 42;
    // signed MIN
    let _ = i32::MIN.min(x);
    let _ = i32::MIN.max(x);
    let _ = x.min(i32::MIN);
    let _ = x.max(i32::MIN);

    let _ = x.min(i32::MIN - 0);
    let _ = x.max(i32::MIN);

    let _ = x.min(i32::MIN - 0);

    // The below cases shouldn't be lint
    let mut min = u32::MAX;
    for _ in 0..1000 {
        min = min.min(random_u32());
    }

    let _ = 2.min(external_consts::MAGIC_NUMBER);
    let _ = 2.max(external_consts::MAGIC_NUMBER);
    let _ = external_consts::MAGIC_NUMBER.min(2);
    let _ = external_consts::MAGIC_NUMBER.max(2);

    let _ = X.min(external_consts::MAGIC_NUMBER);
    let _ = X.max(external_consts::MAGIC_NUMBER);
    let _ = external_consts::MAGIC_NUMBER.min(X);
    let _ = external_consts::MAGIC_NUMBER.max(X);

    let _ = X.max(12);
    let _ = X.min(12);
    let _ = 12.min(X);
    let _ = 12.max(X);
    let _ = (X + 1).max(12);
    let _ = (X + 1).min(12);
    let _ = 12.min(X - 1);
    let _ = 12.max(X - 1);
}
fn random_u32() -> u32 {
    // random number generator
    0
}

struct Issue13191 {
    min: u16,
    max: u16,
}

impl Issue13191 {
    fn new() -> Self {
        Self { min: 0, max: 0 }
    }

    fn min(mut self, value: u16) -> Self {
        self.min = value;
        self
    }

    fn max(mut self, value: u16) -> Self {
        self.max = value;
        self
    }
}

fn issue_13191() {
    // should not fixed
    Issue13191::new().min(0);

    // should not fixed
    Issue13191::new().max(0);
}
