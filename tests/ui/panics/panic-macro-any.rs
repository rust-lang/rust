//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:Box<dyn Any>
//@ ignore-emscripten no processes

#![allow(non_fmt_panics)]

fn main() {
    panic!(Box::new(413) as Box<dyn std::any::Any + Send>);
}
