// Ideally, this would be UB regardless of #[non_exhaustive]. For now,
// at least the semantics don't depend on the crate you're in.
//
// See: rust-lang/rust#147722
#![allow(dead_code)]
#![allow(unreachable_patterns)]

#[repr(u8)]
enum Exhaustive {
    A(u8) = 0,
}

#[repr(u8)]
#[non_exhaustive]
enum NonExhaustive {
    A(u8) = 0,
}

use std::mem::MaybeUninit;

fn main() {
    let buffer: [MaybeUninit<u8>; 2] = [MaybeUninit::uninit(), MaybeUninit::new(0u8)];
    let exh: *const Exhaustive = (&raw const buffer).cast();
    let nexh: *const NonExhaustive = (&raw const buffer).cast();
    unsafe {
        match *exh {
            Exhaustive::A(ref _val) => {}
            _ => {}
        }

        match *nexh { //~ ERROR: memory is uninitialized
            NonExhaustive::A(ref _val) => {}
            _ => {}
        }
    }
}
