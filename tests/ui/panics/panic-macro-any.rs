//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:Box<dyn Any>
//@ needs-subprocess

#![allow(non_fmt_panics)]

fn main() {
    panic!(Box::new(413) as Box<dyn std::any::Any + Send>);
}
