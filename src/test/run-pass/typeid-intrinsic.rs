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

extern crate other1 = "typeid-intrinsic";
extern crate other2 = "typeid-intrinsic2";

use std::hash;
use std::intrinsics;
use std::intrinsics::TypeId;

struct A;
struct Test;

pub fn main() {
    unsafe {
        assert_eq!(intrinsics::type_id::<other1::A>(), other1::id_A());
        assert_eq!(intrinsics::type_id::<other1::B>(), other1::id_B());
        assert_eq!(intrinsics::type_id::<other1::C>(), other1::id_C());
        assert_eq!(intrinsics::type_id::<other1::D>(), other1::id_D());
        assert_eq!(intrinsics::type_id::<other1::E>(), other1::id_E());
        assert_eq!(intrinsics::type_id::<other1::F>(), other1::id_F());
        assert_eq!(intrinsics::type_id::<other1::G>(), other1::id_G());
        assert_eq!(intrinsics::type_id::<other1::H>(), other1::id_H());

        assert_eq!(intrinsics::type_id::<other2::A>(), other2::id_A());
        assert_eq!(intrinsics::type_id::<other2::B>(), other2::id_B());
        assert_eq!(intrinsics::type_id::<other2::C>(), other2::id_C());
        assert_eq!(intrinsics::type_id::<other2::D>(), other2::id_D());
        assert_eq!(intrinsics::type_id::<other2::E>(), other2::id_E());
        assert_eq!(intrinsics::type_id::<other2::F>(), other2::id_F());
        assert_eq!(intrinsics::type_id::<other2::G>(), other2::id_G());
        assert_eq!(intrinsics::type_id::<other2::H>(), other2::id_H());

        assert_eq!(other1::id_F(), other2::id_F());
        assert_eq!(other1::id_G(), other2::id_G());
        assert_eq!(other1::id_H(), other2::id_H());

        assert_eq!(intrinsics::type_id::<int>(), other2::foo::<int>());
        assert_eq!(intrinsics::type_id::<int>(), other1::foo::<int>());
        assert_eq!(other2::foo::<int>(), other1::foo::<int>());
        assert_eq!(intrinsics::type_id::<A>(), other2::foo::<A>());
        assert_eq!(intrinsics::type_id::<A>(), other1::foo::<A>());
        assert_eq!(other2::foo::<A>(), other1::foo::<A>());
    }

    // sanity test of TypeId
    let (a, b, c) = (TypeId::of::<uint>(), TypeId::of::<&'static str>(),
                     TypeId::of::<Test>());
    let (d, e, f) = (TypeId::of::<uint>(), TypeId::of::<&'static str>(),
                     TypeId::of::<Test>());

    assert!(a != b);
    assert!(a != c);
    assert!(b != c);

    assert_eq!(a, d);
    assert_eq!(b, e);
    assert_eq!(c, f);

    // check it has a hash
    let (a, b) = (TypeId::of::<uint>(), TypeId::of::<uint>());

    assert_eq!(hash::hash(&a), hash::hash(&b));
}
