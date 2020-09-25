// run-pass
// check-run-results

#![feature(rustc_private)]

extern crate libc;

fn main() {
    extern "C" fn bye() {
        print!(", world!");
    }
    unsafe { libc::atexit(bye) };
    print!("hello");
}
