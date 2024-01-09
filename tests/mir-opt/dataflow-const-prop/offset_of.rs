// unit-test: DataflowConstProp
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(offset_of)]

use std::marker::PhantomData;
use std::mem::offset_of;

struct Alpha {
    x: u8,
    y: u16,
    z: Beta,
}

struct Beta(u8, u8);

struct Gamma<T> {
    x: u8,
    y: u16,
    _t: T,
}

#[repr(C)]
struct Delta<T> {
    _phantom: PhantomData<T>,
    x: u8,
    y: u16,
}

// EMIT_MIR offset_of.concrete.DataflowConstProp.diff

// CHECK-LABEL: fn concrete
fn concrete() {
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: debug z0 => [[z0:_.*]];
    // CHECK: debug z1 => [[z1:_.*]];

    // CHECK: [[x]] = must_use::<usize>(const 4_usize) -> [return: {{bb[0-9]+}}, unwind continue];
    let x = offset_of!(Alpha, x);

    // CHECK: [[y]] = must_use::<usize>(const 0_usize) -> [return: {{bb[0-9]+}}, unwind continue];
    let y = offset_of!(Alpha, y);

    // CHECK: [[z0]] = must_use::<usize>(const 2_usize) -> [return: {{bb[0-9]+}}, unwind continue];
    let z0 = offset_of!(Alpha, z.0);

    // CHECK: [[z1]] = must_use::<usize>(const 3_usize) -> [return: {{bb[0-9]+}}, unwind continue];
    let z1 = offset_of!(Alpha, z.1);
}

// EMIT_MIR offset_of.generic.DataflowConstProp.diff

// CHECK-LABEL: generic
fn generic<T>() {
    // CHECK: debug gx => [[gx:_.*]];
    // CHECK: debug gy => [[gy:_.*]];
    // CHECK: debug dx => [[dx:_.*]];
    // CHECK: debug dy => [[dy:_.*]];

    // CHECK: [[gx]] = must_use::<usize>(move {{_[0-9]+}}) -> [return: {{bb[0-9]+}}, unwind continue];
    let gx = offset_of!(Gamma<T>, x);

    // CHECK: [[gy]] = must_use::<usize>(move {{_[0-9]+}}) -> [return: {{bb[0-9]+}}, unwind continue];
    let gy = offset_of!(Gamma<T>, y);

    // CHECK: [[dx]] = must_use::<usize>(const 0_usize) -> [return: {{bb[0-9]+}}, unwind continue];
    let dx = offset_of!(Delta<T>, x);

    // CHECK: [[dy]] = must_use::<usize>(const 2_usize) -> [return: {{bb[0-9]+}}, unwind continue];
    let dy = offset_of!(Delta<T>, y);
}

fn main() {
    concrete();
    generic::<()>();
}
