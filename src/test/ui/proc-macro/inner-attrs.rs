// check-pass
// compile-flags: -Z span-debug --error-format human
// aux-build:test-macros.rs

#![feature(custom_inner_attributes)]

#[macro_use]
extern crate test_macros;

#[print_target_and_args(first)]
#[print_target_and_args(second)]
fn foo() {
    #![print_target_and_args(third)]
    #![print_target_and_args(fourth)]
}

fn main() {}
