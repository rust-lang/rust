// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Verifies all possible restrictions for statics values.

// gate-test-drop_types_in_const

#![feature(box_syntax)]

use std::marker;

struct WithDtor;

impl Drop for WithDtor {
    fn drop(&mut self) {}
}

// This enum will be used to test the following rules:
// 1. Variants are safe for static
// 2. Expr calls are allowed as long as they arguments are safe
// 3. Expr calls with unsafe arguments for statics are rejected
enum SafeEnum {
    Variant1,
    Variant2(isize),
    Variant3(WithDtor),
    Variant4(String)
}

// These should be ok
static STATIC1: SafeEnum = SafeEnum::Variant1;
static STATIC2: SafeEnum = SafeEnum::Variant2(0);

// This one should fail
static STATIC3: SafeEnum = SafeEnum::Variant3(WithDtor);
//~^ ERROR destructors in statics are an unstable feature


// This enum will be used to test that variants
// are considered unsafe if their enum type implements
// a destructor.
enum UnsafeEnum {
    Variant5,
    Variant6(isize)
}

impl Drop for UnsafeEnum {
    fn drop(&mut self) {}
}


static STATIC4: UnsafeEnum = UnsafeEnum::Variant5;
//~^ ERROR destructors in statics are an unstable feature
static STATIC5: UnsafeEnum = UnsafeEnum::Variant6(0);
//~^ ERROR destructors in statics are an unstable feature


struct SafeStruct {
    field1: SafeEnum,
    field2: SafeEnum,
}


// Struct fields are safe, hence this static should be safe
static STATIC6: SafeStruct = SafeStruct{field1: SafeEnum::Variant1, field2: SafeEnum::Variant2(0)};

// field2 has an unsafe value, hence this should fail
static STATIC7: SafeStruct = SafeStruct{field1: SafeEnum::Variant1,
                                        field2: SafeEnum::Variant3(WithDtor)};
//~^ ERROR destructors in statics are an unstable feature

// Test variadic constructor for structs. The base struct should be examined
// as well as every field present in the constructor.
// This example shouldn't fail because all the fields are safe.
static STATIC8: SafeStruct = SafeStruct{field1: SafeEnum::Variant1,
                                        ..SafeStruct{field1: SafeEnum::Variant1,
                                                     field2: SafeEnum::Variant1}};

// This example should fail because field1 in the base struct is not safe
static STATIC9: SafeStruct = SafeStruct{field1: SafeEnum::Variant1,
                                        ..SafeStruct{field1: SafeEnum::Variant3(WithDtor),
                                                     field2: SafeEnum::Variant1}};
//~^^ ERROR destructors in statics are an unstable feature

struct UnsafeStruct;

impl Drop for UnsafeStruct {
    fn drop(&mut self) {}
}

// Types with destructors are not allowed for statics
static STATIC10: UnsafeStruct = UnsafeStruct;
//~^ ERROR destructors in statics are an unstable feature

struct MyOwned;

static STATIC11: Box<MyOwned> = box MyOwned;
//~^ ERROR allocations are not allowed in statics

// The following examples test that mutable structs are just forbidden
// to have types with destructors
// These should fail
static mut STATIC12: UnsafeStruct = UnsafeStruct;
//~^ ERROR destructors in statics are an unstable feature
//~^^ ERROR destructors in statics are an unstable feature

static mut STATIC13: SafeStruct = SafeStruct{field1: SafeEnum::Variant1,
//~^ ERROR destructors in statics are an unstable feature
                                             field2: SafeEnum::Variant3(WithDtor)};
//~^ ERROR: destructors in statics are an unstable feature

static mut STATIC14: SafeStruct = SafeStruct {
//~^ ERROR destructors in statics are an unstable feature
    field1: SafeEnum::Variant1,
    field2: SafeEnum::Variant4("str".to_string())
//~^ ERROR calls in statics are limited to constant functions
};

static STATIC15: &'static [Box<MyOwned>] = &[
    box MyOwned, //~ ERROR allocations are not allowed in statics
    box MyOwned, //~ ERROR allocations are not allowed in statics
];

static STATIC16: (&'static Box<MyOwned>, &'static Box<MyOwned>) = (
    &box MyOwned, //~ ERROR allocations are not allowed in statics
    &box MyOwned, //~ ERROR allocations are not allowed in statics
);

static mut STATIC17: SafeEnum = SafeEnum::Variant1;
//~^ ERROR destructors in statics are an unstable feature

static STATIC19: Box<isize> =
    box 3;
//~^ ERROR allocations are not allowed in statics

pub fn main() {
    let y = { static x: Box<isize> = box 3; x };
    //~^ ERROR allocations are not allowed in statics
    //~^^ ERROR cannot move out of static item
}
