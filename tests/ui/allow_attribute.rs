// run-rustfix
#![allow(unused)]
#![warn(clippy::allow_attribute)]
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
