//@ compile-flags: -Zmir-opt-level=0
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(offset_of_enum)]

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

enum Blah {
    A,
    B { x: u8, y: usize },
}

// CHECK-LABEL: fn concrete(
fn concrete() {
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: debug h => [[h:_.*]];
    // CHECK: debug z0 => [[z0:_.*]];
    // CHECK: debug z1 => [[z1:_.*]];

    // CHECK: [[x]] = const concrete::[[const_x:.*]];
    let x = offset_of!(Alpha, x);

    // CHECK: [[y]] = const concrete::[[const_y:.*]];
    let y = offset_of!(Alpha, y);

    // CHECK: [[h]] = const concrete::[[const_h:.*]];
    let h = offset_of!(Blah, B.y);

    // CHECK: [[z0]] = const concrete::[[const_z0:.*]];
    let z0 = offset_of!(Alpha, z.0);

    // CHECK: [[z1]] = const concrete::[[const_z1:.*]];
    let z1 = offset_of!(Alpha, z.1);
}

// CHECK: concrete::[[const_x]]: usize
// CHECK: _0 = offset_of::<Alpha>(const 0_u32, const 0_u32)

// CHECK: concrete::[[const_y]]: usize
// CHECK: _0 = offset_of::<Alpha>(const 0_u32, const 1_u32)

// CHECK: concrete::[[const_h]]: usize
// CHECK: _0 = offset_of::<Blah>(const 1_u32, const 1_u32)

// CHECK: concrete::[[const_z0]]: usize
// CHECK: [[z:_.*]] = offset_of::<Alpha>(const 0_u32, const 2_u32)
// CHECK: [[z0:_.*]] = offset_of::<Beta>(const 0_u32, const 0_u32)
// CHECK: [[sum:_.*]] = AddWithOverflow(copy [[z]], copy [[z0]]);
// CHECK: _0 = move ([[sum]].0: usize);

// CHECK: concrete::[[const_z1]]: usize
// CHECK: [[z:_.*]] = offset_of::<Alpha>(const 0_u32, const 2_u32)
// CHECK: [[z1:_.*]] = offset_of::<Beta>(const 0_u32, const 1_u32)
// CHECK: [[sum:_.*]] = AddWithOverflow(copy [[z]], copy [[z1]]);
// CHECK: _0 = move ([[sum]].0: usize);

// CHECK-LABEL: fn generic(
fn generic<T>() {
    // CHECK: debug gx => [[gx:_.*]];
    // CHECK: debug gy => [[gy:_.*]];
    // CHECK: debug dx => [[dx:_.*]];
    // CHECK: debug dy => [[dy:_.*]];

    // CHECK: [[gx]] = const generic::<T>::[[const_gx:.*]];
    let gx = offset_of!(Gamma<T>, x);

    // CHECK: [[gy]] = const generic::<T>::[[const_gy:.*]];
    let gy = offset_of!(Gamma<T>, y);

    // CHECK: [[dx]] = const generic::<T>::[[const_dx:.*]];
    let dx = offset_of!(Delta<T>, x);

    // CHECK: [[dy]] = const generic::<T>::[[const_dy:.*]];
    let dy = offset_of!(Delta<T>, y);
}

// CHECK: generic::[[const_gx]]: usize
// CHECK: _0 = offset_of::<Gamma<T>>(const 0_u32, const 0_u32)

// CHECK: generic::[[const_gy]]: usize
// CHECK: _0 = offset_of::<Gamma<T>>(const 0_u32, const 1_u32)

// CHECK: generic::[[const_dx]]: usize
// CHECK: _0 = offset_of::<Delta<T>>(const 0_u32, const 1_u32)

// CHECK: generic::[[const_dy]]: usize
// CHECK: _0 = offset_of::<Delta<T>>(const 0_u32, const 2_u32)

fn main() {
    concrete();
    generic::<()>();
}
