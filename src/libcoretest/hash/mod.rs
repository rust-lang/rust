// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use core::kinds::Sized;
use std::mem;

use core::slice::SliceExt;
use core::hash::{Hash, Hasher, Writer};

struct MyWriterHasher;

impl Hasher<MyWriter> for MyWriterHasher {
    fn hash<Sized? T: Hash<MyWriter>>(&self, value: &T) -> u64 {
        let mut state = MyWriter { hash: 0 };
        value.hash(&mut state);
        state.hash
    }
}

struct MyWriter {
    hash: u64,
}

impl Writer for MyWriter {
    // Most things we'll just add up the bytes.
    fn write(&mut self, buf: &[u8]) {
        for byte in buf.iter() {
            self.hash += *byte as u64;
        }
    }
}

#[test]
fn test_writer_hasher() {
    let hasher = MyWriterHasher;

    assert_eq!(hasher.hash(&()), 0);

    assert_eq!(hasher.hash(&5u8), 5);
    assert_eq!(hasher.hash(&5u16), 5);
    assert_eq!(hasher.hash(&5u32), 5);
    assert_eq!(hasher.hash(&5u64), 5);
    assert_eq!(hasher.hash(&5u), 5);

    assert_eq!(hasher.hash(&5i8), 5);
    assert_eq!(hasher.hash(&5i16), 5);
    assert_eq!(hasher.hash(&5i32), 5);
    assert_eq!(hasher.hash(&5i64), 5);
    assert_eq!(hasher.hash(&5i), 5);

    assert_eq!(hasher.hash(&false), 0);
    assert_eq!(hasher.hash(&true), 1);

    assert_eq!(hasher.hash(&'a'), 97);

    let s: &str = "a";
    assert_eq!(hasher.hash(& s), 97 + 0xFF);
    // FIXME (#18283) Enable test
    //let s: Box<str> = box "a";
    //assert_eq!(hasher.hash(& s), 97 + 0xFF);
    let cs: &[u8] = &[1u8, 2u8, 3u8];
    assert_eq!(hasher.hash(& cs), 9);
    let cs: Box<[u8]> = box [1u8, 2u8, 3u8];
    assert_eq!(hasher.hash(& cs), 9);

    // FIXME (#18248) Add tests for hashing Rc<str> and Rc<[T]>

    unsafe {
        let ptr: *const int = mem::transmute(5i);
        assert_eq!(hasher.hash(&ptr), 5);
    }

    unsafe {
        let ptr: *mut int = mem::transmute(5i);
        assert_eq!(hasher.hash(&ptr), 5);
    }
}

struct Custom {
    hash: u64
}

impl Hash<u64> for Custom {
    fn hash(&self, state: &mut u64) {
        *state = self.hash;
    }
}

#[test]
fn test_custom_state() {
    let custom = Custom { hash: 5 };
    let mut state = 0;
    custom.hash(&mut state);
    assert_eq!(state, 5);
}
