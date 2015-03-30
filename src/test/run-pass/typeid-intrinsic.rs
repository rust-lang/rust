// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:typeid-intrinsic.rs
// aux-build:typeid-intrinsic2.rs

// pretty-expanded FIXME #23616

#![feature(hash, core)]

extern crate typeid_intrinsic as other1;
extern crate typeid_intrinsic2 as other2;

use std::hash::{self, SipHasher};
use std::any::TypeId;

struct A;
struct Test;

pub fn main() {
    unsafe {
        assert_eq!(TypeId::of::<other1::A>(), other1::id_A());
        assert_eq!(TypeId::of::<other1::B>(), other1::id_B());
        assert_eq!(TypeId::of::<other1::C>(), other1::id_C());
        assert_eq!(TypeId::of::<other1::D>(), other1::id_D());
        assert_eq!(TypeId::of::<other1::E>(), other1::id_E());
        assert_eq!(TypeId::of::<other1::F>(), other1::id_F());
        assert_eq!(TypeId::of::<other1::G>(), other1::id_G());
        assert_eq!(TypeId::of::<other1::H>(), other1::id_H());

        assert_eq!(TypeId::of::<other2::A>(), other2::id_A());
        assert_eq!(TypeId::of::<other2::B>(), other2::id_B());
        assert_eq!(TypeId::of::<other2::C>(), other2::id_C());
        assert_eq!(TypeId::of::<other2::D>(), other2::id_D());
        assert_eq!(TypeId::of::<other2::E>(), other2::id_E());
        assert_eq!(TypeId::of::<other2::F>(), other2::id_F());
        assert_eq!(TypeId::of::<other2::G>(), other2::id_G());
        assert_eq!(TypeId::of::<other2::H>(), other2::id_H());

        assert_eq!(other1::id_F(), other2::id_F());
        assert_eq!(other1::id_G(), other2::id_G());
        assert_eq!(other1::id_H(), other2::id_H());

        assert_eq!(TypeId::of::<isize>(), other2::foo::<isize>());
        assert_eq!(TypeId::of::<isize>(), other1::foo::<isize>());
        assert_eq!(other2::foo::<isize>(), other1::foo::<isize>());
        assert_eq!(TypeId::of::<A>(), other2::foo::<A>());
        assert_eq!(TypeId::of::<A>(), other1::foo::<A>());
        assert_eq!(other2::foo::<A>(), other1::foo::<A>());
    }

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

    assert_eq!(hash::hash::<TypeId, SipHasher>(&a),
               hash::hash::<TypeId, SipHasher>(&b));
}
