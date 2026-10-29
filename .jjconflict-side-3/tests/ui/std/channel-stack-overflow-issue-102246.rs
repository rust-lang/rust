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
use std::thread::Builder;

const N: usize = 32_768;
const SLOTS: usize = 32;
// Use a stack size that's smaller than N * SLOTS, proving the allocation is on the heap.
//
// The test explicitly specifies the stack size, because not all platforms have the same default
// size.
const STACK_SIZE: usize = (N*SLOTS) - 1;

struct BigStruct {
    _data: [u8; N],
}

fn main() {
    let (sender, receiver) = channel::<BigStruct>();

    let thread1 = Builder::new().stack_size(STACK_SIZE).spawn(move || {
        sender.send(BigStruct { _data: [0u8; N] }).unwrap();
    }).expect("thread1 should spawn successfully");
    thread1.join().unwrap();

    let thread2 = Builder::new().stack_size(STACK_SIZE).spawn(move || {
        for _data in receiver.try_iter() {}
    }).expect("thread2 should spawn successfully");
    thread2.join().unwrap();
}
