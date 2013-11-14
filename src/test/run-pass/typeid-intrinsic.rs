// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast check-fast doesn't like aux-build
// aux-build:typeid-intrinsic.rs
// aux-build:typeid-intrinsic2.rs

extern mod other1(name = "typeid-intrinsic");
extern mod other2(name = "typeid-intrinsic2");

use std::unstable::intrinsics;

struct A;

fn main() {
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
}
