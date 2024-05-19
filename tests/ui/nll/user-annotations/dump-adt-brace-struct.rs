// Unit test for the "user substitutions" that are annotated on each
// node.

//@ compile-flags:-Zverbose-internals

#![allow(warnings)]
#![feature(rustc_attrs)]

struct SomeStruct<T> {
    t: T,
}

#[rustc_dump_user_args]
fn main() {
    SomeStruct { t: 22 }; // Nothing given, no annotation.

    SomeStruct::<_> { t: 22 }; // Nothing interesting given, no annotation.

    SomeStruct::<u32> { t: 22 }; // No lifetime bounds given.

    SomeStruct::<&'static u32> { t: &22 }; //~ ERROR [&'static u32]
}
