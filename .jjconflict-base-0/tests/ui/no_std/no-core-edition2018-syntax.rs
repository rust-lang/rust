//! Test that `#![no_core]` doesn't break modern Rust syntax in edition 2018.
//!
//! When you use `#![no_core]`, you lose the automatic prelude, but you can still
//! get everything back by manually importing `use core::{prelude::v1::*, *}`.
//! This test makes sure that after doing that, things like `for` loops and the
//! `?` operator still work as expected.

//@ run-pass
//@ edition:2018

#![allow(dead_code, unused_imports)]
#![feature(no_core)]
#![no_core]

extern crate core;
extern crate std;
use core::prelude::v1::*;
use core::*;

fn test_for_loop() {
    for _ in &[()] {}
}

fn test_question_mark_operator() -> Option<()> {
    None?
}

fn main() {}
