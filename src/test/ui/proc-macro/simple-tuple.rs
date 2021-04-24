// check-pass
// compile-flags: -Z span-debug --error-format human
// aux-build:test-macros.rs
// edition:2018

#![feature(proc_macro_hygiene)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

fn main() {
    #[print_target_and_args(my_arg)] (
        #![cfg_attr(not(FALSE), allow(unused))]
        1, 2, 3
    );
}
