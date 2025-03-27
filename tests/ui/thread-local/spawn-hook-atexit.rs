// Regression test for https://github.com/rust-lang/rust/issues/138696
//@ only-unix
//@ needs-threads
//@ run-pass

#![feature(rustc_private)]

extern crate libc;

fn main() {
    std::thread::spawn(|| {
        unsafe { libc::atexit(spawn_in_atexit) };
    })
    .join()
    .unwrap();
}

extern "C" fn spawn_in_atexit() {
    std::thread::spawn(|| {
        println!("Thread spawned in atexit");
    })
    .join()
    .unwrap();
}
