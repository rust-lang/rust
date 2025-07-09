//@ run-fail
//@ check-run-results
//@ needs-subprocess

#![allow(non_fmt_panics)]

fn main() {
    panic!(Box::new(612_i64));
}
