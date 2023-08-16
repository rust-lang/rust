// run-fail
//@error-in-other-file:panicked at 'Box<dyn Any>'
//@ignore-target-emscripten no processes

#![allow(non_fmt_panics)]

fn main() {
    panic!(Box::new(612_i64));
}
