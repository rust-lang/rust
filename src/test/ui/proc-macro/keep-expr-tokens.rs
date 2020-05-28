// aux-build:test-macros.rs

#![feature(stmt_expr_attributes)]
#![feature(proc_macro_hygiene)]

extern crate test_macros;

use test_macros::recollect_attr;

fn main() {
    #[test_macros::recollect_attr]
    for item in missing_fn() {} //~ ERROR cannot find

    (#[recollect_attr] #[recollect_attr] ((#[recollect_attr] bad))); //~ ERROR cannot
}
