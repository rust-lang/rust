// skip-filecheck
//@ test-mir-pass: GVN
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

enum Epsilon {
    A(u8, u16),
    B,
    C { c: u32 },
}

enum Zeta<T> {
    A(T, bool),
    B(char),
}

// EMIT_MIR offset_of.concrete.GVN.diff
fn concrete() {
    let x = offset_of!(Alpha, x);
    let y = offset_of!(Alpha, y);
    let z0 = offset_of!(Alpha, z.0);
    let z1 = offset_of!(Alpha, z.1);
    let eA0 = offset_of!(Epsilon, A.0);
    let eA1 = offset_of!(Epsilon, A.1);
    let eC = offset_of!(Epsilon, C.c);
}

// EMIT_MIR offset_of.generic.GVN.diff
fn generic<T>() {
    let gx = offset_of!(Gamma<T>, x);
    let gy = offset_of!(Gamma<T>, y);
    let dx = offset_of!(Delta<T>, x);
    let dy = offset_of!(Delta<T>, y);
    let zA0 = offset_of!(Zeta<T>, A.0);
    let zA1 = offset_of!(Zeta<T>, A.1);
    let zB = offset_of!(Zeta<T>, B.0);
}

fn main() {
    concrete();
    generic::<()>();
}
