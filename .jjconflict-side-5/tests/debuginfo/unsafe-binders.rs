//@ compile-flags: -g
//@ disable-gdb-pretty-printers
//@ ignore-backends: gcc

// Tests that debuginfo is correctly generated for `unsafe<'a> T` binder types.

// === GDB TESTS ===================================================================================

//@ gdb-command:run

//@ gdb-command:whatis binder_i32
//@ gdb-check:type = unsafe &i32

//@ gdb-command:print unwrapped_i32
//@ gdb-check:$1 = 67

//@ gdb-command:whatis no_lifetime
//@ gdb-check:type = unsafe i32

//@ gdb-command:whatis unsafe_binder_tuple
//@ gdb-check:type = unsafe (&i32, &i32)

//@ gdb-command:whatis binder_tuple_ref
//@ gdb-check:type = (&i32, &i32)

//@ gdb-command:whatis binder.inner
//@ gdb-check:type = unsafe &i32

//@ gdb-command:print wrapper.val
//@ gdb-check:$2 = 99

//@ gdb-command:whatis binder_raw
//@ gdb-check:type = unsafe *const i32

//@ gdb-command:print binder_raw_val
//@ gdb-check:$3 = 7

#![feature(unsafe_binders)]
#[expect(incomplete_features)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

struct Wrapper {
    val: i32,
}

struct Binder {
    inner: unsafe<'a> &'a i32,
}

fn main() {
    let x = 67i32;
    let binder_i32: unsafe<'a> &'a i32 = unsafe { wrap_binder!(&x) };
    let unwrapped_i32: i32 = unsafe { *unwrap_binder!(binder_i32) };

    let y = 123i32;
    let no_lifetime: unsafe<> i32 = unsafe { wrap_binder!(y) };

    let unsafe_binder_tuple: unsafe<'a> (&'a i32, &'a i32) = unsafe {
        wrap_binder!((&114i32, &514i32))
    };
    let binder_tuple_ref: (&i32, &i32) = unsafe { unwrap_binder!(unsafe_binder_tuple) };

    let val = 99i32;
    let binder = Binder { inner: unsafe { wrap_binder!(&val) } };
    let wrapper = Wrapper { val: unsafe { *unwrap_binder!(binder.inner) } };

    let z = 7i32;
    let raw: *const i32 = &z;
    let binder_raw: unsafe<'a> *const i32 = unsafe { wrap_binder!(raw) };
    let binder_raw_val: i32 = unsafe { *unwrap_binder!(binder_raw) };

    gugugaga(); // #break
}

fn gugugaga() { () }
