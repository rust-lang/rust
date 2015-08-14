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
        let mut ids = vec!();
        ids.push(TypeId::of::<FooNil>());
        ids.push(TypeId::of::<FooBool>());
        ids.push(TypeId::of::<FooInt>());
        ids.push(TypeId::of::<FooUint>());
        ids.push(TypeId::of::<FooFloat>());
        ids.push(TypeId::of::<FooEnum>());
        ids.push(TypeId::of::<FooUniq>());
        ids.push(TypeId::of::<FooPtr>());
        ids.push(TypeId::of::<&'static FooTrait>());
        ids.push(TypeId::of::<FooStruct>());
        ids.push(TypeId::of::<FooTuple>());
        ids
    }

    // Tests ty_nil
    pub type FooNil = ();

    // Skipping ty_bot

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

    // For TyStr, what kind of string should I use? &'static str? String? Raw str?

    // Tests TyEnum
    pub enum FooEnum {
        VarA(usize),
        VarB(usize, usize)
    }

    // Tests TyBox (of u8)
    pub type FooUniq = Box<u8>;

    // As with TyStr, what type should be used for TyArray?

    // Tests TyRawPtr
    pub type FooPtr = *const u8;

    // Skipping TyRef

    // Skipping TyBareFn (how do you get a bare function type, rather than proc or closure?)

    // Tests TyTrait
    pub trait FooTrait {
        fn foo_method(&self) -> usize;
    }

    // Tests TyStruct
    pub struct FooStruct {
        pub pub_foo_field: usize,
        foo_field: usize
    }

    // Tests TyTuple
    pub type FooTuple = (u8, i8, bool);

    // Skipping ty_param

    // Skipping ty_self

    // Skipping ty_self

    // Skipping TyInfer

    // Skipping TyError
}
