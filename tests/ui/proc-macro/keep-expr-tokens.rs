//@ proc-macro: test-macros.rs
//@ compile-flags: -Z span-debug

#![feature(stmt_expr_attributes)]
#![feature(proc_macro_hygiene)]
#![feature(rustc_attrs)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

extern crate test_macros;

use test_macros::recollect_attr;

fn main() {
    #[test_macros::recollect_attr]
    for item in missing_fn() {} //~ ERROR cannot find

    (#[recollect_attr] #[recollect_attr] ((#[recollect_attr] bad))); //~ ERROR cannot

    #[test_macros::print_attr]
    #[rustc_dummy]
    { 1 +1; } // Don't change the weird spacing of the '+'
}
