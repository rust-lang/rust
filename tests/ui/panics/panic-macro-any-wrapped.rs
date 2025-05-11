//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:Box<dyn Any>
//@ needs-subprocess

#![allow(non_fmt_panics)]

fn main() {
    panic!(Box::new(612_i64));
}
