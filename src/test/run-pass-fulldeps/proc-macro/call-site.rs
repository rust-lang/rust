// aux-build:call-site.rs
// ignore-stage1

#![feature(proc_macro_non_items)]

extern crate call_site;
use call_site::*;

fn main() {
    let x1 = 10;
    call_site::check!(let x2 = x1;);
    let x6 = x5;
}
