//@ proc-macro: test-macros.rs
//@ check-pass
//@ compile-flags: -Z span-debug

#![feature(custom_inner_attributes)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

extern crate test_macros;
use test_macros::{print_target_and_args};

struct Foo<const V: bool>;
trait Bar<const V: bool> {}

#[print_target_and_args(first_outer)]
#[print_target_and_args(second_outer)]
impl Bar<{1 > 0}> for Foo<{true}> {
    #![print_target_and_args(first_inner)]
    #![print_target_and_args(second_inner)]
}

fn main() {}
