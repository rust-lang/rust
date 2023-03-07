// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

fn foo(_: &[&str]) {}

fn bad(a: &str, b: &str) {
    foo(&[a, b]);
}

fn good(a: &str, b: &str) {
    foo(&[a, b]);
}

fn main() {}
