#![allow(dead_code)]

enum E {
    A,
}

struct S {
    field1: i32 = 42, //~ ERROR default values on fields are experimental
    field2: E = E::A, //~ ERROR default values on fields are experimental
    field3: i32 = 1 + 2, //~ ERROR default values on fields are experimental
    field4: i32 = { 1 + 2 }, //~ ERROR default values on fields are experimental
    field5: E = foo(42), //~ ERROR default values on fields are experimental
    field6: E = { foo(42) }, //~ ERROR default values on fields are experimental
}

struct S1 {
    field1: i32 //~ ERROR expected `,`, or `}`, found `field2`
    field2: E //~ ERROR expected `,`, or `}`, found `field3`
    field3: i32 = 1 + 2, //~ ERROR default values on fields are experimental
    field4: i32 = { 1 + 2 }, //~ ERROR default values on fields are experimental
    field5: E = foo(42), //~ ERROR default values on fields are experimental
    field6: E = { foo(42) }, //~ ERROR default values on fields are experimental
}

struct S2 {
    field1 = i32, //~ ERROR expected `:`, found `=`
    field2; E, //~ ERROR expected `:`, found `;`
}

const fn foo(_: i32) -> E {
    E::A
}

fn main() {}
