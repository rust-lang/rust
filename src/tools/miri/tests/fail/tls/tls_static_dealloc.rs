//! Ensure that thread-local statics get deallocated when the thread dies.

#![feature(thread_local)]
// FIXME: Use `SyncUnsafeCell` instead of allowing `static_mut_refs` lint
#![allow(static_mut_refs)]

#[thread_local]
static mut TLS: u8 = 0;

struct SendRaw(*const u8);
unsafe impl Send for SendRaw {}

fn main() {
    unsafe {
        let dangling_ptr = std::thread::spawn(|| SendRaw(&TLS as *const u8)).join().unwrap();
        let _val = *dangling_ptr.0; //~ ERROR: has been freed
    }
}
