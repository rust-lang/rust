// run-rustfix
#![allow(unused)]
#![warn(clippy::allow_attributes)]
#![feature(lint_reasons)]

fn main() {}

// Using clippy::needless_borrow just as a placeholder, it isn't relevant.

// Should lint
#[allow(dead_code)]
struct T1;

struct T2; // Should not lint
#[deny(clippy::needless_borrow)] // Should not lint
struct T3;
#[warn(clippy::needless_borrow)] // Should not lint
struct T4;
// `panic = "unwind"` should always be true
#[cfg_attr(panic = "unwind", allow(dead_code))]
struct CfgT;

fn ignore_inner_attr() {
    #![allow(unused)] // Should not lint
}
