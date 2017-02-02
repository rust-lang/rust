// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod sip;

use std::hash::{Hash, Hasher};
use std::default::Default;

struct MyHasher {
    hash: u64,
}

impl Default for MyHasher {
    fn default() -> MyHasher {
        MyHasher { hash: 0 }
    }
}

impl Hasher for MyHasher {
    fn write(&mut self, buf: &[u8]) {
        for byte in buf {
            self.hash += *byte as u64;
        }
    }
    fn finish(&self) -> u64 { self.hash }
}


#[test]
fn test_writer_hasher() {
    fn hash<T: Hash>(t: &T) -> u64 {
        let mut s = MyHasher { hash: 0 };
        t.hash(&mut s);
        s.finish()
    }

    assert_eq!(hash(&()), 0);

    assert_eq!(hash(&5_u8), 5);
    assert_eq!(hash(&5_u16), 5);
    assert_eq!(hash(&5_u32), 5);
    assert_eq!(hash(&5_u64), 5);
    assert_eq!(hash(&5_usize), 5);

    assert_eq!(hash(&5_i8), 5);
    assert_eq!(hash(&5_i16), 5);
    assert_eq!(hash(&5_i32), 5);
    assert_eq!(hash(&5_i64), 5);
    assert_eq!(hash(&5_isize), 5);

    assert_eq!(hash(&false), 0);
    assert_eq!(hash(&true), 1);

    assert_eq!(hash(&'a'), 97);

    let s: &str = "a";
    assert_eq!(hash(& s), 97 + 0xFF);
    let s: Box<str> = String::from("a").into_boxed_str();
    assert_eq!(hash(& s), 97 + 0xFF);
    let cs: &[u8] = &[1, 2, 3];
    assert_eq!(hash(& cs), 9);
    let cs: Box<[u8]> = Box::new([1, 2, 3]);
    assert_eq!(hash(& cs), 9);

    // FIXME (#18248) Add tests for hashing Rc<str> and Rc<[T]>

    let ptr = 5_usize as *const i32;
    assert_eq!(hash(&ptr), 5);

    let ptr = 5_usize as *mut i32;
    assert_eq!(hash(&ptr), 5);
}

struct Custom { hash: u64 }
struct CustomHasher { output: u64 }

impl Hasher for CustomHasher {
    fn finish(&self) -> u64 { self.output }
    fn write(&mut self, _: &[u8]) { panic!() }
    fn write_u64(&mut self, data: u64) { self.output = data; }
}

impl Default for CustomHasher {
    fn default() -> CustomHasher {
        CustomHasher { output: 0 }
    }
}

impl Hash for Custom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

#[test]
fn test_custom_state() {
    fn hash<T: Hash>(t: &T) -> u64 {
        let mut c = CustomHasher { output: 0 };
        t.hash(&mut c);
        c.finish()
    }

    assert_eq!(hash(&Custom { hash: 5 }), 5);
}
