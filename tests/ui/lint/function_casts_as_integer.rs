//@ run-rustfix

#![deny(function_casts_as_integer)]
#![allow(unused_variables, dead_code)] // For the rustfix-ed code.

fn foo() {}

enum MyEnum {
    Variant(u32),
}

struct MyStruct(u32);

fn main() {
    let x = foo as usize; //~ ERROR: function_casts_as_integer
    let x = String::len as usize; //~ ERROR: function_casts_as_integer
    let x = MyEnum::Variant as usize; //~ ERROR: function_casts_as_integer
    let x = MyStruct as usize; //~ ERROR: function_casts_as_integer
    // Ok.
    let x = foo as fn() as usize;
    let x = foo as *const () as usize;
}
