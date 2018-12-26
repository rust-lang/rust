#![allow(deprecated)]
// aux-build:typeid-intrinsic-aux1.rs
// aux-build:typeid-intrinsic-aux2.rs

#![feature(core_intrinsics)]

extern crate typeid_intrinsic_aux1 as other1;
extern crate typeid_intrinsic_aux2 as other2;

use std::hash::{SipHasher, Hasher, Hash};
use std::any::TypeId;

struct A;
struct Test;

pub fn main() {
    assert_eq!(TypeId::of::<other1::A>(), other1::id_A());
    assert_eq!(TypeId::of::<other1::B>(), other1::id_B());
    assert_eq!(TypeId::of::<other1::C>(), other1::id_C());
    assert_eq!(TypeId::of::<other1::D>(), other1::id_D());
    assert_eq!(TypeId::of::<other1::E>(), other1::id_E());
    assert_eq!(TypeId::of::<other1::F>(), other1::id_F());
    assert_eq!(TypeId::of::<other1::G>(), other1::id_G());
    assert_eq!(TypeId::of::<other1::H>(), other1::id_H());
    assert_eq!(TypeId::of::<other1::I>(), other1::id_I());

    assert_eq!(TypeId::of::<other2::A>(), other2::id_A());
    assert_eq!(TypeId::of::<other2::B>(), other2::id_B());
    assert_eq!(TypeId::of::<other2::C>(), other2::id_C());
    assert_eq!(TypeId::of::<other2::D>(), other2::id_D());
    assert_eq!(TypeId::of::<other2::E>(), other2::id_E());
    assert_eq!(TypeId::of::<other2::F>(), other2::id_F());
    assert_eq!(TypeId::of::<other2::G>(), other2::id_G());
    assert_eq!(TypeId::of::<other2::H>(), other2::id_H());
    assert_eq!(TypeId::of::<other1::I>(), other2::id_I());

    assert_eq!(other1::id_F(), other2::id_F());
    assert_eq!(other1::id_G(), other2::id_G());
    assert_eq!(other1::id_H(), other2::id_H());
    assert_eq!(other1::id_I(), other2::id_I());

    assert_eq!(TypeId::of::<isize>(), other2::foo::<isize>());
    assert_eq!(TypeId::of::<isize>(), other1::foo::<isize>());
    assert_eq!(other2::foo::<isize>(), other1::foo::<isize>());
    assert_eq!(TypeId::of::<A>(), other2::foo::<A>());
    assert_eq!(TypeId::of::<A>(), other1::foo::<A>());
    assert_eq!(other2::foo::<A>(), other1::foo::<A>());

    // sanity test of TypeId
    let (a, b, c) = (TypeId::of::<usize>(), TypeId::of::<&'static str>(),
                     TypeId::of::<Test>());
    let (d, e, f) = (TypeId::of::<usize>(), TypeId::of::<&'static str>(),
                     TypeId::of::<Test>());

    assert!(a != b);
    assert!(a != c);
    assert!(b != c);

    assert_eq!(a, d);
    assert_eq!(b, e);
    assert_eq!(c, f);

    // check it has a hash
    let (a, b) = (TypeId::of::<usize>(), TypeId::of::<usize>());

    let mut s1 = SipHasher::new();
    a.hash(&mut s1);
    let mut s2 = SipHasher::new();
    b.hash(&mut s2);

    assert_eq!(s1.finish(), s2.finish());

    // Check projections

    assert_eq!(TypeId::of::<other1::I32Iterator>(), other1::id_i32_iterator());
    assert_eq!(TypeId::of::<other1::U32Iterator>(), other1::id_u32_iterator());
    assert_eq!(other1::id_i32_iterator(), other2::id_i32_iterator());
    assert_eq!(other1::id_u32_iterator(), other2::id_u32_iterator());
    assert!(other1::id_i32_iterator() != other1::id_u32_iterator());
    assert!(TypeId::of::<other1::I32Iterator>() != TypeId::of::<other1::U32Iterator>());

    // Check fn pointer against collisions
    assert!(TypeId::of::<fn(fn(A) -> A) -> A>() !=
            TypeId::of::<fn(fn() -> A, A) -> A>());
}
