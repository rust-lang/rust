//@ run-pass
//@ needs-threads
// This time we're testing repeatedly going up and down both stacks to
// make sure the stack pointers are maintained properly in both
// directions

use std::thread;

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_call(
        cb: extern "C" fn(u64) -> u64,
        data: u64,
    ) -> u64;
}

extern "C" fn cb(data: u64) -> u64 {
    if data == 1 { data } else { count(data - 1) + count(data - 1) }
}

fn count(n: u64) -> u64 {
    unsafe {
        println!("n = {}", n);
        rust_dbg_call(cb, n)
    }
}

pub fn main() {
    // Make sure we're on a thread with small Rust stacks (main currently
    // has a large stack)
    thread::spawn(move || {
        let result = count(12);
        println!("result = {}", result);
        assert_eq!(result, 2048);
    })
    .join().unwrap();
}
