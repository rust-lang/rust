//@compile-flags: -Zmiri-disable-isolation -Zmiri-deterministic-concurrency
//@ignore-target: windows # No libc env support on Windows

use std::{env, thread};

fn main() {
    let t = thread::spawn(|| unsafe {
        // Access the environment in another thread without taking the env lock.
        // This represents some C code that queries the environment.
        libc::getenv(c"TZ".as_ptr()); //~ERROR: Data race detected
    });
    // Meanwhile, the main thread uses the "safe" Rust env accessor.
    env::set_var("MY_RUST_VAR", "Ferris");

    t.join().unwrap();
}
