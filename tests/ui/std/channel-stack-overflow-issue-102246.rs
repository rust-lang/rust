//@ run-pass
//@ needs-threads
//@ compile-flags: -Copt-level=0

// The channel's `Block::new` was causing a stack overflow because it held 32 item slots, which is
// 1MiB for this test's `BigStruct` -- instantiated on the stack before moving to `Box::new`.
//
// That block is now initialized directly on the heap.
//
// Ref: https://github.com/rust-lang/rust/issues/102246

use std::sync::mpsc::channel;
use std::thread;

const N: usize = 32_768;
struct BigStruct {
    _data: [u8; N],
}

fn main() {
    let (sender, receiver) = channel::<BigStruct>();

    let thread1 = thread::spawn(move || {
        sender.send(BigStruct { _data: [0u8; N] }).unwrap();
    });

    thread1.join().unwrap();
    for _data in receiver.try_iter() {}
}
