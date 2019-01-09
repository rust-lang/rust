// run-pass
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(deprecated)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(overflowing_literals)]

use std::hash::{Hash, SipHasher, Hasher};
use std::mem::size_of;

#[derive(Hash)]
struct Person {
    id: usize,
    name: String,
    phone: usize,
}

// test for hygiene name collisions
#[derive(Hash)] struct __H__H;
#[derive(Hash)] enum Collision<__H> { __H { __H__H: __H } }

#[derive(Hash)]
enum E { A=1, B }

fn hash<T: Hash>(t: &T) -> u64 {
    let mut s = SipHasher::new_with_keys(0, 0);
    t.hash(&mut s);
    s.finish()
}

struct FakeHasher<'a>(&'a mut Vec<u8>);
impl<'a> Hasher for FakeHasher<'a> {
    fn finish(&self) -> u64 {
        unimplemented!()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.extend(bytes);
    }
}

fn fake_hash<A: Hash>(v: &mut Vec<u8>, a: A) {
    a.hash(&mut FakeHasher(v));
}

fn main() {
    let person1 = Person {
        id: 5,
        name: "Janet".to_string(),
        phone: 555_666_7777
    };
    let person2 = Person {
        id: 5,
        name: "Bob".to_string(),
        phone: 555_666_7777
    };
    assert_eq!(hash(&person1), hash(&person1));
    assert!(hash(&person1) != hash(&person2));

    // test #21714
    let mut va = vec![];
    let mut vb = vec![];
    fake_hash(&mut va, E::A);
    fake_hash(&mut vb, E::B);
    assert!(va != vb);

    // issue #39137: single variant enum hash should not hash discriminant
    #[derive(Hash)]
    enum SingleVariantEnum {
        A(u8),
    }
    let mut v = vec![];
    fake_hash(&mut v, SingleVariantEnum::A(17));
    assert_eq!(vec![17], v);
}
