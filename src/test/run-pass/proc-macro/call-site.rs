#![allow(unused_variables)]
#![allow(unused_imports)]
// aux-build:call-site.rs

#![feature(proc_macro_hygiene)]

extern crate call_site;
use call_site::*;

fn main() {
    let x1 = 10;
    call_site::check!(let x2 = x1;);
    let x6 = x5;
}
