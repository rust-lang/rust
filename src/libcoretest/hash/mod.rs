// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;
use std::hash::{Hash, Hasher, Writer};
use std::default::Default;

struct MyHasher {
    hash: u64,
}

impl Default for MyHasher {
    fn default() -> MyHasher {
        MyHasher { hash: 0 }
    }
}

impl Writer for MyHasher {
    // Most things we'll just add up the bytes.
    fn write(&mut self, buf: &[u8]) {
        for byte in buf.iter() {
            self.hash += *byte as u64;
        }
    }
}

impl Hasher for MyHasher {
    type Output = u64;
    fn reset(&mut self) { self.hash = 0; }
    fn finish(&self) -> u64 { self.hash }
}


#[test]
fn test_writer_hasher() {
    fn hash<T: Hash<MyHasher>>(t: &T) -> u64 {
        ::std::hash::hash::<_, MyHasher>(t)
    }

    assert_eq!(hash(&()), 0);

    assert_eq!(hash(&5u8), 5);
    assert_eq!(hash(&5u16), 5);
    assert_eq!(hash(&5u32), 5);
    assert_eq!(hash(&5u64), 5);
    assert_eq!(hash(&5u), 5);

    assert_eq!(hash(&5i8), 5);
    assert_eq!(hash(&5i16), 5);
    assert_eq!(hash(&5i32), 5);
    assert_eq!(hash(&5i64), 5);
    assert_eq!(hash(&5i), 5);

    assert_eq!(hash(&false), 0);
    assert_eq!(hash(&true), 1);

    assert_eq!(hash(&'a'), 97);

    let s: &str = "a";
    assert_eq!(hash(& s), 97 + 0xFF);
    // FIXME (#18283) Enable test
    //let s: Box<str> = box "a";
    //assert_eq!(hasher.hash(& s), 97 + 0xFF);
    let cs: &[u8] = &[1u8, 2u8, 3u8];
    assert_eq!(hash(& cs), 9);
    let cs: Box<[u8]> = box [1u8, 2u8, 3u8];
    assert_eq!(hash(& cs), 9);

    // FIXME (#18248) Add tests for hashing Rc<str> and Rc<[T]>

    unsafe {
        let ptr: *const int = mem::transmute(5i);
        assert_eq!(hash(&ptr), 5);
    }

    unsafe {
        let ptr: *mut int = mem::transmute(5i);
        assert_eq!(hash(&ptr), 5);
    }
}

struct Custom { hash: u64 }
struct CustomHasher { output: u64 }

impl Hasher for CustomHasher {
    type Output = u64;
    fn reset(&mut self) { self.output = 0; }
    fn finish(&self) -> u64 { self.output }
}

impl Default for CustomHasher {
    fn default() -> CustomHasher {
        CustomHasher { output: 0 }
    }
}

impl Hash<CustomHasher> for Custom {
    fn hash(&self, state: &mut CustomHasher) {
        state.output = self.hash;
    }
}

#[test]
fn test_custom_state() {
    fn hash<T: Hash<CustomHasher>>(t: &T) -> u64 {
        ::std::hash::hash::<_, CustomHasher>(t)
    }

    assert_eq!(hash(&Custom { hash: 5 }), 5);
}
