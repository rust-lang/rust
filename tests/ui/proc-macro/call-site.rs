//@ check-pass
//@ aux-build:call-site.rs

extern crate call_site;

fn main() {
    let x1 = 10;
    call_site::check!(let x2 = x1;);
    let x6 = x5;
}
