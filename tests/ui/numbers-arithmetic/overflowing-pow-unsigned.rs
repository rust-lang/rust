//@ run-fail
//@ check-run-results:thread 'main' panicked
//@ check-run-results:attempt to multiply with overflow
//@ ignore-emscripten no processes
//@ compile-flags: -C debug-assertions

fn main() {
    let _x = 2u32.pow(1024);
}
