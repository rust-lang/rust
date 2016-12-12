// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core)]

pub mod testtypes {
    use std::any::TypeId;

    pub fn type_ids() -> Vec<TypeId> {
        vec![
            TypeId::of::<FooBool>(),
            TypeId::of::<FooInt>(),
            TypeId::of::<FooUint>(),
            TypeId::of::<FooFloat>(),
            TypeId::of::<FooStr>(),
            TypeId::of::<FooArray>(),
            TypeId::of::<FooSlice>(),
            TypeId::of::<FooBox>(),
            TypeId::of::<FooPtr>(),
            TypeId::of::<FooRef>(),
            TypeId::of::<FooFnPtr>(),
            TypeId::of::<FooNil>(),
            TypeId::of::<FooTuple>(),
            TypeId::of::<FooTrait>(),
            TypeId::of::<FooStruct>(),
            TypeId::of::<FooEnum>()
        ]
    }

    // Tests TyBool
    pub type FooBool = bool;

    // Tests TyChar
    pub type FooChar = char;

    // Tests TyInt (does not test all variants of IntTy)
    pub type FooInt = isize;

    // Tests TyUint (does not test all variants of UintTy)
    pub type FooUint = usize;

    // Tests TyFloat (does not test all variants of FloatTy)
    pub type FooFloat = f64;

    // Tests TyStr
    pub type FooStr = str;

    // Tests TyArray
    pub type FooArray = [u8; 1];

    // Tests TySlice
    pub type FooSlice = [u8];

    // Tests TyBox (of u8)
    pub type FooBox = Box<u8>;

    // Tests TyRawPtr
    pub type FooPtr = *const u8;

    // Tests TyRef
    pub type FooRef = &'static u8;

    // Tests TyFnPtr
    pub type FooFnPtr = fn(u8) -> bool;

    // Tests TyDynamic
    pub trait FooTrait {
        fn foo_method(&self) -> usize;
    }

    // Tests struct
    pub struct FooStruct {
        pub pub_foo_field: usize,
        foo_field: usize
    }

    // Tests enum
    pub enum FooEnum {
        VarA(usize),
        VarB(usize, usize)
    }

    // Tests TyTuple
    pub type FooNil = ();
    pub type FooTuple = (u8, i8, bool);

    // Skipping TyParam

    // Skipping TyInfer

    // Skipping TyError
}
