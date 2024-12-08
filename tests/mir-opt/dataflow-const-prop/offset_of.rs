//@ test-mir-pass: DataflowConstProp
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

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

// CHECK-LABEL: fn concrete(
fn concrete() {
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: debug z0 => [[z0:_.*]];
    // CHECK: debug z1 => [[z1:_.*]];

    // CHECK: [[x]] = const 4_usize
    let x = offset_of!(Alpha, x);

    // CHECK: [[y]] = const 0_usize
    let y = offset_of!(Alpha, y);

    // CHECK: [[z0]] = const 2_usize
    let z0 = offset_of!(Alpha, z.0);

    // CHECK: [[z1]] = const 3_usize
    let z1 = offset_of!(Alpha, z.1);
}

// EMIT_MIR offset_of.generic.DataflowConstProp.diff

// CHECK-LABEL: fn generic(
fn generic<T>() {
    // CHECK: debug gx => [[gx:_.*]];
    // CHECK: debug gy => [[gy:_.*]];
    // CHECK: debug dx => [[dx:_.*]];
    // CHECK: debug dy => [[dy:_.*]];

    // CHECK: [[gx]] = OffsetOf(Gamma<T>, [(0, 0)]);
    let gx = offset_of!(Gamma<T>, x);

    // CHECK: [[gy]] = OffsetOf(Gamma<T>, [(0, 1)]);
    let gy = offset_of!(Gamma<T>, y);

    // CHECK: [[dx]] = const 0_usize
    let dx = offset_of!(Delta<T>, x);

    // CHECK: [[dy]] = const 2_usize
    let dy = offset_of!(Delta<T>, y);
}

fn main() {
    concrete();
    generic::<()>();
}
