// Verifies all possible restrictions for statics values.

#![allow(warnings)]

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
    Variant4(String),
}

// These should be ok
static STATIC1: SafeEnum = SafeEnum::Variant1;
static STATIC2: SafeEnum = SafeEnum::Variant2(0);
static STATIC3: SafeEnum = SafeEnum::Variant3(WithDtor);

enum UnsafeEnum {
    Variant5,
    Variant6(isize),
}

impl Drop for UnsafeEnum {
    fn drop(&mut self) {}
}

static STATIC4: UnsafeEnum = UnsafeEnum::Variant5;
static STATIC5: UnsafeEnum = UnsafeEnum::Variant6(0);

struct SafeStruct {
    field1: SafeEnum,
    field2: SafeEnum,
}

// Struct fields are safe, hence this static should be safe
static STATIC6: SafeStruct =
    SafeStruct { field1: SafeEnum::Variant1, field2: SafeEnum::Variant2(0) };

static STATIC7: SafeStruct =
    SafeStruct { field1: SafeEnum::Variant1, field2: SafeEnum::Variant3(WithDtor) };

// Test variadic constructor for structs. The base struct should be examined
// as well as every field present in the constructor.
// This example shouldn't fail because all the fields are safe.
static STATIC8: SafeStruct = SafeStruct {
    field1: SafeEnum::Variant1,
    ..SafeStruct { field1: SafeEnum::Variant1, field2: SafeEnum::Variant1 }
};

// This example should fail because field1 in the base struct is not safe
static STATIC9: SafeStruct = SafeStruct {
    field1: SafeEnum::Variant1,
    ..SafeStruct {
        //~^ ERROR destructor of
        field1: SafeEnum::Variant3(WithDtor),
        field2: SafeEnum::Variant1,
    }
};

struct UnsafeStruct;

impl Drop for UnsafeStruct {
    fn drop(&mut self) {}
}

static STATIC10: UnsafeStruct = UnsafeStruct;

struct MyOwned;

static STATIC11: Vec<MyOwned> = vec![MyOwned];
//~^ ERROR allocations are not allowed in statics
//~^^ ERROR cannot call non-const

static mut STATIC12: UnsafeStruct = UnsafeStruct;

static mut STATIC13: SafeStruct =
    SafeStruct { field1: SafeEnum::Variant1, field2: SafeEnum::Variant3(WithDtor) };

static mut STATIC14: SafeStruct = SafeStruct {
    field1: SafeEnum::Variant1,
    field2: SafeEnum::Variant4("str".to_string()), //~ ERROR cannot call non-const fn
};

static STATIC15: &'static [Vec<MyOwned>] = &[
    vec![MyOwned], //~ ERROR allocations are not allowed in statics
    //~^ ERROR cannot call non-const
    vec![MyOwned], //~ ERROR allocations are not allowed in statics
                   //~^ ERROR cannot call non-const
];

static STATIC16: (&'static Vec<MyOwned>, &'static Vec<MyOwned>) = (
    &vec![MyOwned], //~ ERROR allocations are not allowed in statics
    //~^ ERROR cannot call non-const
    &vec![MyOwned], //~ ERROR allocations are not allowed in statics
                    //~^ ERROR cannot call non-const
);

static mut STATIC17: SafeEnum = SafeEnum::Variant1;

static STATIC19: Vec<isize> = vec![3];
//~^ ERROR allocations are not allowed in statics
//~^^ ERROR cannot call non-const

pub fn main() {
    let y = {
        static x: Vec<isize> = vec![3]; //~ ERROR allocations are not allowed in statics
        //~^ ERROR cannot call non-const
        x
        //~^ ERROR cannot move out of static
    };
}
