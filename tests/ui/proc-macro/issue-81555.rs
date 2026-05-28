//@ check-pass
//@ proc-macro: test-macros.rs
#![feature(stmt_expr_attributes, proc_macro_hygiene)]

extern crate test_macros;

use test_macros::identity_attr;

#[identity_attr]
fn main() {
    let _x;
    let y = ();
    #[identity_attr]
    _x = y;
}
