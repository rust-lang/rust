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
    // CHECK [[a:_.*]] = (_1.0: u8);
    // CHECK _.* = access::<u8>(move [[a]]) -> [return: bb1, unwind: bb5];
    access(foo.a);
    // CHECK [[b:_.*]] = ((_1.1: Foo::{anon_adt#0}).0: i8);
    // CHECK _.* = access::<i8>(move [[b]]) -> [return: bb2, unwind: bb5];
    access(foo.b);
    // CHECK [[c:_.*]] = ((_1.1: Foo::{anon_adt#0}).1: bool);
    // CHECK _.* = access::<bool>(move [[c]]) -> [return: bb3, unwind: bb5];
    access(foo.c);
    // CHECK [[d:_.*]] = (((_1.2: Foo::{anon_adt#1}).0: Foo::{anon_adt#1}::{anon_adt#0}).0: [u8; 1]);
    // CHECK _.* = access::<[u8; 1]>(move [[d]]) -> [return: bb4, unwind: bb5];
    access(foo.d);
}

// CHECK-LABEL: fn bar(
fn bar(bar: Bar) {
    unsafe {
        // CHECK [[a:_.*]] = (_1.0: u8);
        // CHECK _.* = access::<u8>(move [[a]]) -> [return: bb1, unwind: bb5];
        access(bar.a);
        // CHECK [[b:_.*]] = ((_1.1: Bar::{anon_adt#0}).0: i8);
        // CHECK _.* = access::<i8>(move [[b]]) -> [return: bb2, unwind: bb5];
        access(bar.b);
        // CHECK [[c:_.*]] = ((_1.1: Bar::{anon_adt#0}).1: bool);
        // CHECK _.* = access::<bool>(move [[c]]) -> [return: bb3, unwind: bb5];
        access(bar.c);
        // CHECK [[d:_.*]] = (((_1.2: Bar::{anon_adt#1}).0: Bar::{anon_adt#1}::{anon_adt#0}).0: [u8; 1]);
        // CHECK _.* = access::<[u8; 1]>(move [[d]]) -> [return: bb4, unwind: bb5];
        access(bar.d);
    }
}


fn main() {}
