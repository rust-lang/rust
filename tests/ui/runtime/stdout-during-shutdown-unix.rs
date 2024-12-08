//@ run-pass
//@ check-run-results
//@ ignore-emscripten
//@ only-unix

// Emscripten doesn't flush its own stdout buffers on exit, which would fail
// this test. So this test is disabled on this platform.
// See https://emscripten.org/docs/getting_started/FAQ.html#what-does-exiting-the-runtime-mean-why-don-t-atexit-s-run

#![feature(rustc_private)]

extern crate libc;

fn main() {
    extern "C" fn bye() {
        print!(", world!");
    }
    unsafe { libc::atexit(bye) };
    print!("hello");
}
