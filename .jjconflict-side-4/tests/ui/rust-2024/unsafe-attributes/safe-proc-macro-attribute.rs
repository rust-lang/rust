//! Anti-regression test for `#[safe]` proc-macro attribute.

//@ revisions: unknown_attr proc_macro_attr
//@[proc_macro_attr] proc-macro: safe_attr.rs
//@[proc_macro_attr] check-pass

#![warn(unsafe_attr_outside_unsafe)]

#[cfg(proc_macro_attr)]
extern crate safe_attr;
#[cfg(proc_macro_attr)]
use safe_attr::safe;

#[safe]
//[unknown_attr]~^ ERROR cannot find attribute `safe` in this scope
fn foo() {}

#[safe(no_mangle)]
//[unknown_attr]~^ ERROR cannot find attribute `safe` in this scope
fn bar() {}

fn main() {}
