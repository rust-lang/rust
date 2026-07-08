// Like single_variant_uninit.rs, but with a non_exhaustive enum, as the generated MIR used to
// differ between these cases.
//
// See: rust-lang/rust#147722
//
// This UB should be detected even with validation disabled.
//@compile-flags: -Zmiri-disable-validation
#![allow(dead_code)]
#![allow(unreachable_patterns)]

#[repr(u8)]
#[non_exhaustive]
enum NonExhaustive {
    A(u8) = 0,
}

use std::mem::MaybeUninit;

fn main() {
    let buffer: [MaybeUninit<u8>; 2] = [MaybeUninit::uninit(), MaybeUninit::new(0u8)];
    let nexh: *const NonExhaustive = (&raw const buffer).cast();
    unsafe {
        match *nexh {
            //~^ ERROR: memory is uninitialized
            NonExhaustive::A(ref _val) => {}
            _ => {}
        }
    }
}
