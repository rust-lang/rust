//@ skip-filecheck
//@ test-mir-pass: GVN
//@ compile-flags: -C panic=abort

#![feature(freeze)]

use std::marker::Freeze;

#[derive(Clone, Copy)]
pub union U<'a> {
    pub i: i32,
    pub f: &'a i32,
}

// EMIT_MIR indirect_ref.indirect_union.GVN.diff
#[inline(never)]
pub fn indirect_union(x: &U<'_>, out: fn(U<'_>), clobber: fn()) {
    let y1 = *x;
    out(y1);
    clobber();
    let y2 = *x;
    out(y2);
}

// EMIT_MIR indirect_ref.indirect_deref_1.GVN.diff
#[inline(never)]
pub fn indirect_deref_1(x: &(&i32,), out: fn(i32), clobber: fn()) {
    let y1 = *x;
    out(*(y1).0);
    clobber();
    let y2 = *x;
    out(*(y2).0);
}

// EMIT_MIR indirect_ref.indirect_deref_2.GVN.diff
#[inline(never)]
pub fn indirect_deref_2(x: &(&i32,), out: fn(i32), clobber: fn()) {
    out(*(*x).0);
    clobber();
    out(*(*x).0);
}

// EMIT_MIR indirect_ref.indirect_ref_1.GVN.diff
#[inline(never)]
pub fn indirect_ref_1(x: &(&i32,), out: fn(&i32), clobber: fn()) {
    out((*x).0);
    clobber();
    out((*x).0);
}

// EMIT_MIR indirect_ref.indirect_ref_2.GVN.diff
#[inline(never)]
pub fn indirect_ref_2(x: &(&i32,), out: fn(&i32), clobber: fn()) {
    let y1 = *x;
    out((y1).0);
    clobber();
    let y2 = *x;
    out((y2).0);
}

// EMIT_MIR indirect_ref.indirect_ref_t_1.GVN.diff
#[inline(never)]
pub fn indirect_ref_t_1<T: Copy + Freeze>(x: &(T,), out: fn(T), clobber: fn()) {
    out((*x).0);
    clobber();
    out((*x).0);
}

// EMIT_MIR indirect_ref.indirect_ref_t_2.GVN.diff
#[inline(never)]
pub fn indirect_ref_t_2<T: Copy + Freeze>(x: &(T,), out: fn(T), clobber: fn()) {
    let y1 = *x;
    out((y1).0);
    clobber();
    let y2 = *x;
    out((y2).0);
}

#[derive(Clone, Copy)]
pub struct Adt<'a>(pub &'a i32);

// EMIT_MIR indirect_ref.indirect_adt.GVN.diff
#[inline(never)]
pub fn indirect_adt(x: &Adt<'_>, out: fn(Adt), clobber: fn()) {
    let y1 = *x;
    out(y1);
    clobber();
    let y2 = *x;
    out(y2);
}

static mut DATA: i32 = 0;

fn main() {
    let ptr = &raw mut DATA;
    let nested_shr: &(&i32,) = unsafe { &*(&raw const ptr as *const (&i32,)) };
    let u = U { f: nested_shr.0 };
    indirect_union(&u, |x| assert_eq!(unsafe { *x.f }, unsafe { DATA }), || unsafe { DATA += 1 });
    indirect_deref_1(nested_shr, |x| assert_eq!(x, unsafe { DATA }), || unsafe { DATA += 1 });
    indirect_deref_2(nested_shr, |x| assert_eq!(x, unsafe { DATA }), || unsafe { DATA += 1 });
    indirect_ref_1(nested_shr, |&x| assert_eq!(x, unsafe { DATA }), || unsafe { DATA += 1 });
    indirect_ref_2(nested_shr, |&x| assert_eq!(x, unsafe { DATA }), || unsafe { DATA += 1 });
    indirect_ref_t_1(nested_shr, |&x| assert_eq!(x, unsafe { DATA }), || unsafe { DATA += 1 });
    indirect_ref_t_2(nested_shr, |&x| assert_eq!(x, unsafe { DATA }), || unsafe { DATA += 1 });
    let adt = Adt(nested_shr.0);
    indirect_adt(&adt, |x| assert_eq!(*x.0, unsafe { DATA }), || unsafe { DATA += 1 });
}
