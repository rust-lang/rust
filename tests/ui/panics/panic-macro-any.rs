//@ run-fail
//@ check-run-results:panicked
//@ check-run-results:Box<dyn Any>
//@ ignore-emscripten no processes

#![allow(non_fmt_panics)]

fn main() {
    panic!(Box::new(413) as Box<dyn std::any::Any + Send>);
}
