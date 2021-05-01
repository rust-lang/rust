// run-rustfix
#![allow(dead_code)]

enum E {
    A,
}

struct S {
    field1: i32 = 42, //~ ERROR default values on `struct` fields aren't supported
    field2: E = E::A, //~ ERROR default values on `struct` fields aren't supported
    field3: i32 = 1 + 2, //~ ERROR default values on `struct` fields aren't supported
    field4: i32 = { 1 + 2 }, //~ ERROR default values on `struct` fields aren't supported
    field5: E = foo(42), //~ ERROR default values on `struct` fields aren't supported
    field6: E = { foo(42) }, //~ ERROR default values on `struct` fields aren't supported
}

struct S1 {
    field1: i32 //~ ERROR expected `,`, or `}`, found `field2`
    field2: E //~ ERROR expected `,`, or `}`, found `field3`
    field3: i32 = 1 + 2, //~ ERROR default values on `struct` fields aren't supported
    field4: i32 = { 1 + 2 }, //~ ERROR default values on `struct` fields aren't supported
    field5: E = foo(42), //~ ERROR default values on `struct` fields aren't supported
    field6: E = { foo(42) }, //~ ERROR default values on `struct` fields aren't supported
}

struct S2 {
    field1 = i32, //~ ERROR expected `:`, found `=`
    field2; E, //~ ERROR expected `:`, found `;`
}

const fn foo(_: i32) -> E {
    E::A
}

fn main() {}
