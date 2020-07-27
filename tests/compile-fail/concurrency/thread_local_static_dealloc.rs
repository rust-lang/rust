// ignore-windows: Concurrency on Windows is not supported yet.

//! Ensure that thread-local statics get deallocated when the thread dies.

#![feature(thread_local)]

#[thread_local]
static mut TLS: u8 = 0;

fn main() { unsafe {
    let dangling_ptr = std::thread::spawn(|| &TLS as *const u8 as usize).join().unwrap();
    let _val = *(dangling_ptr as *const u8); //~ ERROR dereferenced after this allocation got freed
} }
