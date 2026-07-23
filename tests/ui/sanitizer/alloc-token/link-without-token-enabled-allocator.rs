// Verifies that programs without a token-enabled memory allocator degrade gracefully (i.e.,
// programs remain correct, without heap partitioning).
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Ctarget-feature=-crt-static -Cprefer-dynamic=off -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token
//@ run-pass

#![allow(dead_code)]
#![feature(allocator_api)]
#![feature(alloc_with_token)]

use std::alloc::Allocator;

pub struct Buffer {
    data: [u8; 4096],
}

pub struct Node {
    next: *mut Node,
    value: u64,
}

fn main() {
    let buffer = Box::new(Buffer { data: [1; 4096] });
    let node = Box::new(Node { next: core::ptr::null_mut(), value: 1 });
    assert_eq!(buffer.data[0], 1);
    assert_eq!(node.value, 1);
    let layout = std::alloc::Layout::new::<Buffer>();
    let ptr = std::alloc::System.allocate_with_token(layout, 0).unwrap();
    unsafe { std::alloc::System.deallocate(ptr.cast(), layout) };

    std::hint::black_box((buffer, node));
}
