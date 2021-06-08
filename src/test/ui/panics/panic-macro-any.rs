// run-fail
// error-pattern:panicked at 'Box<dyn Any>'
// ignore-emscripten no processes

#![feature(box_syntax)]
#![allow(non_fmt_panic)]

fn main() {
    panic!(box 413 as Box<dyn std::any::Any + Send>);
}
