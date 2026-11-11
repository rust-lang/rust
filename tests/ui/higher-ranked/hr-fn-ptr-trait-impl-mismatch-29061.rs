//! Regression test for <https://github.com/rust-lang/rust/issues/29061>.
//!
//! A trait implemented for a higher-ranked fn pointer is not implemented for a fn pointer
//! with a specific lifetime, and vice versa. The errors now spell out which form the impl
//! applies to instead of just saying the bound is unsatisfied.

//@ edition: 2024

#![allow(dead_code)]

fn x(_: &()) {}

trait HR {}
impl HR for fn(&()) {}
fn hr<T: HR>(_: T) {}

trait NotHR {}
impl<'a> NotHR for fn(&'a ()) {}
fn not_hr<T: NotHR>(_: T) {}

fn a<'a>() {
    let not_hr_func: fn(&'a ()) = x;
    let hr_func: fn(&()) = x;
    let hr_func2: for<'b> fn(&'b ()) = x;
    hr(not_hr_func);
    //~^ ERROR implementation of `HR` is not general enough
    not_hr(hr_func);
    //~^ ERROR implementation of `NotHR` is not general enough
    not_hr(hr_func2);
    //~^ ERROR implementation of `NotHR` is not general enough
}

fn main() {}
