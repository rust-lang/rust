// run-fail
// error-pattern:panicked at 'Box<dyn Any>'
// ignore-emscripten no processes

#![allow(non_fmt_panic)]

fn main() {
    panic!(Box::new(612_i64));
}
