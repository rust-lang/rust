#![warn(clippy::uninit_vec)]

use std::mem::MaybeUninit;

fn main() {
    // with_capacity() -> set_len() should be detected
    let mut vec: Vec<u8> = Vec::with_capacity(1000);
    unsafe {
        vec.set_len(200);
    }

    // reserve() -> set_len() should be detected
    vec.reserve(1000);
    unsafe {
        vec.set_len(200);
    }

    // test when both calls are enclosed in the same unsafe block
    unsafe {
        let mut vec: Vec<u8> = Vec::with_capacity(1000);
        vec.set_len(200);

        vec.reserve(1000);
        vec.set_len(200);
    }

    // MaybeUninit-wrapped types should not be detected
    let mut vec: Vec<MaybeUninit<u8>> = Vec::with_capacity(1000);
    unsafe {
        vec.set_len(200);
    }
}
