// run-fail
//@error-in-other-file:thread 'main' panicked at 'attempt to multiply with overflow'
//@ignore-target-emscripten no processes
//@compile-flags: -C debug-assertions

fn main() {
    let _x = 2i32.pow(1024);
}
