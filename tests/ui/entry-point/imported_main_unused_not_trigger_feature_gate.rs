//@ check-pass
#![feature(rustc_attrs)]

#[rustc_main]
fn actual_main() {}

mod foo {
    pub(crate) fn something() {}
}

use foo::something as main;
