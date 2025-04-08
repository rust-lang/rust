#![feature(large_assignments)]
#![deny(large_assignments)]
#![move_size_limit = "1000"]

//! Tests that with `-Zinline-mir`, we do NOT get an error that points to the
//! implementation of `UnsafeCell` since that is not actionable by the user:
//!
//! ```text
//! error: moving 9999 bytes
//!   --> /rustc/FAKE_PREFIX/library/core/src/cell.rs:2054:9
//!    |
//!    = note: value moved from here
//! ```
//!
//! We want the diagnostics to point to the relevant user code.

//@ build-fail
//@ compile-flags: -Zmir-opt-level=1 -Zinline-mir

pub fn main() {
    let data = [10u8; 9999];
    let cell = std::cell::UnsafeCell::new(data); //~ ERROR large_assignments
    std::hint::black_box(cell);
}
