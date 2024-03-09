// Tests the correct handling of unnamed fields within structs and unions marked with #[repr(C)].

// EMIT_MIR field_access.foo.SimplifyCfg-initial.after.mir
// EMIT_MIR field_access.bar.SimplifyCfg-initial.after.mir

#![allow(incomplete_features)]
#![feature(unnamed_fields)]

#[repr(C)]
struct Foo {
    a: u8,
    _: struct {
        b: i8,
        c: bool,
    },
    _: struct {
        _: struct {
            d: [u8; 1],
        }
    }
}

#[repr(C)]
union Bar {
    a: u8,
    _: union {
        b: i8,
        c: bool,
    },
    _: union {
        _: union {
            d: [u8; 1],
        }
    }
}


fn access<T>(_: T) {}

// CHECK-LABEL: fn foo(
fn foo(foo: Foo) {
    // CHECK _3 = (_1.0: u8);
    // CHECK _2 = access::<u8>(move _3) -> [return: bb1, unwind: bb5];
    access(foo.a);
    // CHECK _5 = ((_1.1: Foo::{anon_adt#0}).0: i8);
    // CHECK _4 = access::<i8>(move _5) -> [return: bb2, unwind: bb5];
    access(foo.b);
    // CHECK _7 = ((_1.1: Foo::{anon_adt#0}).1: bool);
    // CHECK _6 = access::<bool>(move _7) -> [return: bb3, unwind: bb5];
    access(foo.c);
    // CHECK _9 = (((_1.2: Foo::{anon_adt#1}).0: Foo::{anon_adt#1}::{anon_adt#0}).0: [u8; 1]);
    // CHECK _8 = access::<[u8; 1]>(move _9) -> [return: bb4, unwind: bb5];
    access(foo.d);
}

// CHECK-LABEL: fn bar(
fn bar(bar: Bar) {
    unsafe {
        // CHECK _3 = (_1.0: u8);
        // CHECK _2 = access::<u8>(move _3) -> [return: bb1, unwind: bb5];
        access(bar.a);
        // CHECK _5 = ((_1.1: Bar::{anon_adt#0}).0: i8);
        // CHECK _4 = access::<i8>(move _5) -> [return: bb2, unwind: bb5];
        access(bar.b);
        // CHECK _7 = ((_1.1: Bar::{anon_adt#0}).1: bool);
        // CHECK _6 = access::<bool>(move _7) -> [return: bb3, unwind: bb5];
        access(bar.c);
        // CHECK _9 = (((_1.2: Bar::{anon_adt#1}).0: Bar::{anon_adt#1}::{anon_adt#0}).0: [u8; 1]);
        // CHECK _8 = access::<[u8; 1]>(move _9) -> [return: bb4, unwind: bb5];
        access(bar.d);
    }
}


fn main() {}
