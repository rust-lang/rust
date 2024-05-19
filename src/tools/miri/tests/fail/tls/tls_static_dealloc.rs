//! Ensure that thread-local statics get deallocated when the thread dies.

#![feature(thread_local)]

use std::ptr::addr_of;

#[thread_local]
static mut TLS: u8 = 0;

struct SendRaw(*const u8);
unsafe impl Send for SendRaw {}

fn main() {
    unsafe {
        let dangling_ptr = std::thread::spawn(|| SendRaw(addr_of!(TLS))).join().unwrap();
        let _val = *dangling_ptr.0; //~ ERROR: has been freed
    }
}
