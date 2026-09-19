// Compiler:
//
// Run-time:
//   status: 0

// Checks that the `#[linkage]` flavours another object file is allowed to override are emitted as
// weak symbols, by linking against `tests/c/weak_function_linkage.c`, which defines the same
// symbols strongly.

#![feature(linkage, no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

#[linkage = "weak"]
#[no_mangle]
extern "C" fn weak_function() -> i32 {
    0
}

// `_odr` promises every definition of the symbol is equivalent, which lets a backend call this body
// instead of the one in the C file. They spell the same value for that reason.
#[linkage = "weak_odr"]
#[no_mangle]
extern "C" fn weak_odr_function() -> i32 {
    2
}

#[linkage = "linkonce"]
#[no_mangle]
extern "C" fn linkonce_function() -> i32 {
    0
}

#[linkage = "linkonce_odr"]
#[no_mangle]
extern "C" fn linkonce_odr_function() -> i32 {
    4
}

// `#[linkage = "common"]` is absent on purpose: a common symbol is `SHN_COMMON`, which the object
// format only allows for objects, so no backend can give a function that linkage.

// Not overridden by the C side: the definition here is the one that runs.
#[linkage = "weak"]
#[no_mangle]
extern "C" fn only_weak_function() -> i32 {
    6
}

// The real definition is the one in the C file; a backend may call it or emit an equivalent copy of
// this body, so both spell the same value.
#[linkage = "available_externally"]
#[no_mangle]
extern "C" fn available_externally_function() -> i32 {
    7
}

// GCC drops `weak` from a function that is also `inline`: a backend that keeps the hint emits this
// as an ordinary global symbol and clashes with the C definition. rustc lints the hint as ignored
// on a function with an explicit `#[linkage]`, hence the `allow`.
#[linkage = "weak"]
#[inline]
#[no_mangle]
#[allow(unused_attributes)]
extern "C" fn weak_inline_function() -> i32 {
    0
}

extern "C" {
    fn c_call_all() -> i32;
}

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    let result = unsafe { c_call_all() };
    if result != 0 {
        return result;
    }

    if weak_function() != 1 {
        return 1;
    }
    if weak_odr_function() != 2 {
        return 2;
    }
    if linkonce_function() != 3 {
        return 3;
    }
    if linkonce_odr_function() != 4 {
        return 4;
    }
    if only_weak_function() != 6 {
        return 6;
    }
    if available_externally_function() != 7 {
        return 7;
    }
    if weak_inline_function() != 8 {
        return 8;
    }
    0
}
