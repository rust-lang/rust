extern crate foobar;
extern crate foobar_macro;

use foobar::*;
use foobar_macro::*;

a_proc_macro!(); // no

#[an_attr_macro]
fn a() {
    f(); // no
}

#[an_attr_macro(with_span)]
fn b() {
    f(); // yes
}

fn c() {
    a_rules_macro!(f()); // yes
}

fn d() {
    a_rules_macro!(()); // no
}

fn main() {}
