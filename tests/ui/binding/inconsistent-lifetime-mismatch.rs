//@ run-pass
#![allow(dead_code)]

fn foo(_: &[&str]) {}

fn bad(a: &str, b: &str) {
    foo(&[a, b]);
}

fn good(a: &str, b: &str) {
    foo(&[a, b]);
}

fn main() {}
