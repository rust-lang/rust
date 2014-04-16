// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod testtypes {
    use std::intrinsics::TypeId;

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
        ids.push(TypeId::of::<FooClosure>());
        ids.push(TypeId::of::<&'static FooTrait>());
        ids.push(TypeId::of::<FooStruct>());
        ids.push(TypeId::of::<FooTuple>());
        ids
    }

    // Tests ty_nil
    pub type FooNil = ();

    // Skipping ty_bot

    // Tests ty_bool
    pub type FooBool = bool;

    // Tests ty_char
    pub type FooChar = char;

    // Tests ty_int (does not test all variants of IntTy)
    pub type FooInt = int;

    // Tests ty_uint (does not test all variants of UintTy)
    pub type FooUint = uint;

    // Tests ty_float (does not test all variants of FloatTy)
    pub type FooFloat = f64;

    // For ty_str, what kind of string should I use? &'static str? ~str? Raw str?

    // Tests ty_enum
    pub enum FooEnum {
        VarA(uint),
        VarB(uint, uint)
    }

    // Skipping ty_box

    // Tests ty_uniq (of u8)
    pub type FooUniq = ~u8;

    // As with ty_str, what type should be used for ty_vec?

    // Tests ty_ptr
    pub type FooPtr = *u8;

    // Skipping ty_rptr

    // Skipping ty_bare_fn (how do you get a bare function type, rather than proc or closure?)

    // Tests ty_closure (does not test all types of closures)
    pub type FooClosure = |arg: u8|: 'static -> u8;

    // Tests ty_trait
    pub trait FooTrait {
        fn foo_method(&self) -> uint;
        fn foo_static_method() -> uint;
    }

    // Tests ty_struct
    pub struct FooStruct {
        pub pub_foo_field: uint,
        foo_field: uint
    }

    // Tests ty_tup
    pub type FooTuple = (u8, i8, bool);

    // Skipping ty_param

    // Skipping ty_self

    // Skipping ty_self

    // Skipping ty_infer

    // Skipping ty_err
}
