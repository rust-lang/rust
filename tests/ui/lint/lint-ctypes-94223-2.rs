#![crate_type = "lib"]
#![deny(improper_ctypes_definitions)]

// Test is split in two files since some warnings arise during WF-checking
// and some later, so we can't get them all in the same file.

pub fn bad(f: extern "C" fn([u8])) {}
//~^ ERROR the size for values of type `[u8]` cannot be known at compilation time

pub fn bad_twice(f: Result<extern "C" fn([u8]), extern "C" fn([u8])>) {}
//~^ ERROR the size for values of type `[u8]` cannot be known at compilation time

struct BadStruct(extern "C" fn([u8]));
//~^ ERROR the size for values of type `[u8]` cannot be known at compilation time

enum BadEnum {
    A(extern "C" fn([u8])),
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}

enum BadUnion {
    A(extern "C" fn([u8])),
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}

pub trait FooTrait {
    type FooType;
}

pub struct FfiUnsafe;

#[allow(improper_ctypes_definitions)]
extern "C" fn f(_: FfiUnsafe) {
    unimplemented!()
}
