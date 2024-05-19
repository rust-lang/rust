//@ compile-flags: -g -Cstrip=none -Cforce-frame-pointers=yes

#![crate_type = "cdylib"]
#![crate_type = "rlib"]

#![allow(improper_ctypes_definitions)]

type Pos = (&'static str, u32);

macro_rules! pos {
    () => {
        (file!(), line!())
    };
}

#[no_mangle]
pub extern "C" fn foo(outer: Pos, inner: fn(Pos, Pos)) {
    inner(outer, pos!());
}
