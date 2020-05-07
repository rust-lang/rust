// run-fail
// error-pattern:meep
// ignore-emscripten no processes

#![feature(box_syntax)]

fn f(_a: isize, _b: isize, _c: Box<isize>) {
    panic!("moop");
}

fn main() {
    f(1, panic!("meep"), box 42);
}
