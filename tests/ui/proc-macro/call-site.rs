//@ check-pass
//@ proc-macro: call-site.rs
//@ ignore-backends: gcc

extern crate call_site;

fn main() {
    let x1 = 10;
    call_site::check!(let x2 = x1;);
    let x6 = x5;
}
