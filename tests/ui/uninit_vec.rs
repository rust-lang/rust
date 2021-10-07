#![warn(clippy::uninit_vec)]

use std::mem::MaybeUninit;

#[derive(Default)]
struct MyVec {
    vec: Vec<u8>,
}

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

    // new() -> set_len() should be detected
    let mut vec: Vec<u8> = Vec::new();
    unsafe {
        vec.set_len(200);
    }

    // default() -> set_len() should be detected
    let mut vec: Vec<u8> = Default::default();
    unsafe {
        vec.set_len(200);
    }

    let mut vec: Vec<u8> = Vec::default();
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

    let mut vec: Vec<u8> = Vec::with_capacity(1000);
    unsafe {
        // test the case where there are other statements in the following unsafe block
        vec.set_len(200);
        assert!(vec.len() == 200);
    }

    // handle vec stored in the field of a struct
    let mut my_vec = MyVec::default();
    my_vec.vec.reserve(1000);
    unsafe {
        my_vec.vec.set_len(200);
    }

    my_vec.vec = Vec::with_capacity(1000);
    unsafe {
        my_vec.vec.set_len(200);
    }

    // Test `#[allow(...)]` attributes on inner unsafe block (shouldn't trigger)
    let mut vec: Vec<u8> = Vec::with_capacity(1000);
    #[allow(clippy::uninit_vec)]
    unsafe {
        vec.set_len(200);
    }

    // MaybeUninit-wrapped types should not be detected
    unsafe {
        let mut vec: Vec<MaybeUninit<u8>> = Vec::with_capacity(1000);
        vec.set_len(200);

        let mut vec: Vec<(MaybeUninit<u8>, MaybeUninit<bool>)> = Vec::with_capacity(1000);
        vec.set_len(200);

        let mut vec: Vec<(MaybeUninit<u8>, [MaybeUninit<bool>; 2])> = Vec::with_capacity(1000);
        vec.set_len(200);
    }

    // known false negative
    let mut vec1: Vec<u8> = Vec::with_capacity(1000);
    let mut vec2: Vec<u8> = Vec::with_capacity(1000);
    unsafe {
        vec1.set_len(200);
        vec2.set_len(200);
    }
}
