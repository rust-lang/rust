//@ run-fail
//@ check-run-results
//@ needs-subprocess

#![allow(non_fmt_panics)]

fn main() {
    panic!(Box::new(413) as Box<dyn std::any::Any + Send>);
}
