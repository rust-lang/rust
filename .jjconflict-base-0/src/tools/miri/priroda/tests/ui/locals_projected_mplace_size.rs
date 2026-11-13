//@ compile-flags: -Zmir-opt-level=0

#![allow(internal_features)]
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// Regression test for projected values: each `item.<field>` row must render
// only that field's own value/layout, and the memory-backed `item.target` row
// must not continue through the rest of the allocation backing `Envelope`.
#[repr(C)]
struct Payload {
    a: u16,
    b: u8,
    c: u32,
    d: u8,
}

#[repr(C)]
struct Envelope {
    prefix: u8,
    target: Payload,
    trailer: u64,
    checksum: u16,
}

#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn projected_struct(item: Envelope) {
    mir! {
        debug prefix => item.prefix;
        debug target => item.target;
        debug trailer => item.trailer;
        debug checksum => item.checksum;
        {
            Return()
        }
    }
}

fn main() {
    projected_struct(Envelope {
        prefix: 0xaa,
        target: Payload { a: 0x1122, b: 0x33, c: 0x44556677, d: 0x88 },
        trailer: 0x99aabbccddeeff00,
        checksum: 0x1234,
    });
}
