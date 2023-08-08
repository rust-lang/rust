// unit-test: ConstProp
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

// EMIT_MIR offset_of.concrete.ConstProp.diff
fn concrete() {
    let x = offset_of!(Alpha, x);
    let y = offset_of!(Alpha, y);
    let z0 = offset_of!(Alpha, z.0);
    let z1 = offset_of!(Alpha, z.1);
}

// EMIT_MIR offset_of.generic.ConstProp.diff
fn generic<T>() {
    let gx = offset_of!(Gamma<T>, x);
    let gy = offset_of!(Gamma<T>, y);
    let dx = offset_of!(Delta<T>, x);
    let dy = offset_of!(Delta<T>, y);
}

fn main() {
    concrete();
    generic::<()>();
}
