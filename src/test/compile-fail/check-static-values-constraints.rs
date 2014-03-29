// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(managed_boxes)]

// Verifies all possible restrictions for static items values.

use std::kinds::marker;

struct WithDtor;

impl Drop for WithDtor {
    fn drop(&mut self) {}
}

// This enum will be used to test the following rules:
// 1. Variants are safe for static
// 2. Expr calls are allowed as long as they arguments are safe
// 3. Expr calls with unsafe arguments for static items are rejected
enum SafeEnum {
    Variant1,
    Variant2(int),
    Variant3(WithDtor),
    Variant4(~str)
}

// These should be ok
static STATIC1: SafeEnum = Variant1;
static STATIC2: SafeEnum = Variant2(0);

// This one should fail
static STATIC3: SafeEnum = Variant3(WithDtor);
//~^ ERROR static items are not allowed to have destructors


// This enum will be used to test that variants
// are considered unsafe if their enum type implements
// a destructor.
enum UnsafeEnum {
    Variant5,
    Variant6(int)
}

impl Drop for UnsafeEnum {
    fn drop(&mut self) {}
}


static STATIC4: UnsafeEnum = Variant5;
//~^ ERROR static items are not allowed to have destructors
static STATIC5: UnsafeEnum = Variant6(0);
//~^ ERROR static items are not allowed to have destructors


struct SafeStruct {
    field1: SafeEnum,
    field2: SafeEnum,
}


// Struct fields are safe, hence this static should be safe
static STATIC6: SafeStruct = SafeStruct{field1: Variant1, field2: Variant2(0)};

// field2 has an unsafe value, hence this should fail
static STATIC7: SafeStruct = SafeStruct{field1: Variant1, field2: Variant3(WithDtor)};
//~^ ERROR static items are not allowed to have destructors

// Test variadic constructor for structs. The base struct should be examined
// as well as every field persent in the constructor.
// This example shouldn't fail because all the fields are safe.
static STATIC8: SafeStruct = SafeStruct{field1: Variant1,
                                        ..SafeStruct{field1: Variant1, field2: Variant1}};

// This example should fail because field1 in the base struct is not safe
static STATIC9: SafeStruct = SafeStruct{field1: Variant1,
                                        ..SafeStruct{field1: Variant3(WithDtor), field2: Variant1}};
//~^ ERROR static items are not allowed to have destructors

struct UnsafeStruct;

impl Drop for UnsafeStruct {
    fn drop(&mut self) {}
}

// Types with destructors are not allowed for statics
static STATIC10: UnsafeStruct = UnsafeStruct;
//~^ ERROR static items are not allowed to have destructor

static STATIC11: ~str = ~"Owned pointers are not allowed either";
//~^ ERROR static items are not allowed to have owned pointers

// The following examples test that mutable structs are just forbidden
// to have types with destructors
// These should fail
static mut STATIC12: UnsafeStruct = UnsafeStruct;
//~^ ERROR mutable static items are not allowed to have destructors

static mut STATIC13: SafeStruct = SafeStruct{field1: Variant1, field2: Variant3(WithDtor)};
//~^ ERROR mutable static items are not allowed to have destructors

static mut STATIC14: SafeStruct = SafeStruct{field1: Variant1, field2: Variant4(~"str")};
//~^ ERROR mutable static items are not allowed to have destructors

static STATIC15: &'static [~str] = &'static [~"str", ~"str"];
//~^ ERROR static items are not allowed to have owned pointers
//~^^ ERROR static items are not allowed to have owned pointers

static STATIC16: (~str, ~str) = (~"str", ~"str");
//~^ ERROR static items are not allowed to have owned pointers
//~^^ ERROR static items are not allowed to have owned pointers

static mut STATIC17: SafeEnum = Variant1;
//~^ ERROR mutable static items are not allowed to have destructors

static STATIC18: @SafeStruct = @SafeStruct{field1: Variant1, field2: Variant2(0)};
//~^ ERROR static items are not allowed to have managed pointers

static STATIC19: ~int = box 3;
//~^ ERROR static items are not allowed to have owned pointers

pub fn main() {
    let y = { static x: ~int = ~3; x };
    //~^ ERROR static items are not allowed to have owned pointers
}
