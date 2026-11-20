//@ check-pass
//@aux-build:once_cell.rs
//@aux-build:lazy_static.rs

#![warn(clippy::non_std_lazy_statics)]
#![no_std]

use lazy_static::lazy_static;
use once_cell::sync::Lazy;

fn main() {}

static LAZY_FOO: Lazy<usize> = Lazy::new(|| 42);
static LAZY_BAR: Lazy<i32> = Lazy::new(|| {
    let x: i32 = 0;
    x.saturating_add(100)
});

lazy_static! {
    static ref LAZY_BAZ: f64 = 12.159 * 548;
}
