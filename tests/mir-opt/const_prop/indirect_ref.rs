//! Regression test for <https://github.com/rust-lang/rust/issues/155884>.
//! Nested shared references may be NOT read-only.
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
    // CHECK-LABEL: fn indirect_union
    // CHECK: out => [[out:_.*]];
    // CHECK: [[y1:_.*]] = copy (*_1);
    // CHECK: copy [[out]](copy [[y1]]) -> [
    // CHECK: [[y2:_.*]] = copy (*_1);
    // CHECK: copy [[out]](copy [[y2]]) -> [
    let y1 = *x;
    out(y1);
    clobber();
    let y2 = *x;
    out(y2);
}

// EMIT_MIR indirect_ref.indirect_deref_1.GVN.diff
#[inline(never)]
pub fn indirect_deref_1(x: &(&i32,), out: fn(i32), clobber: fn()) {
    // CHECK-LABEL: fn indirect_deref_1
    // CHECK: out => [[out:_.*]];
    // CHECK: [[y1:_.*]] = copy (*_1);
    // CHECK: [[y1_0:_.*]] = no_retag copy ([[y1]].0: &i32);
    // CHECK: [[deref_y1_0:_.*]] = copy (*[[y1_0]]);
    // CHECK: copy [[out]](move [[deref_y1_0]]) -> [
    // CHECK: [[y2:_.*]] = copy (*_1);
    // CHECK: [[y2_0:_.*]] = no_retag copy ([[y2]].0: &i32);
    // CHECK: [[deref_y2_0:_.*]] = copy (*[[y2_0]]);
    // CHECK: copy [[out]](move [[deref_y2_0]]) -> [
    let y1 = *x;
    out(*(y1).0);
    clobber();
    let y2 = *x;
    out(*(y2).0);
}

// EMIT_MIR indirect_ref.indirect_deref_2.GVN.diff
#[inline(never)]
pub fn indirect_deref_2(x: &(&i32,), out: fn(i32), clobber: fn()) {
    // CHECK-LABEL: fn indirect_deref_2
    // CHECK: out => [[out:_.*]];
    // CHECK: [[y1_0:_.*]] = no_retag copy ((*_1).0: &i32);
    // CHECK: [[deref_y1_0:_.*]] = copy (*[[y1_0]]);
    // CHECK: copy [[out]](move [[deref_y1_0]]) -> [
    // CHECK: [[y2_0:_.*]] = no_retag copy ((*_1).0: &i32);
    // CHECK: [[deref_y2_0:_.*]] = copy (*[[y2_0]]);
    // CHECK: copy [[out]](move [[deref_y2_0]]) -> [
    out(*(*x).0);
    clobber();
    out(*(*x).0);
}

// EMIT_MIR indirect_ref.indirect_ref_1.GVN.diff
#[inline(never)]
pub fn indirect_ref_1(x: &(&i32,), out: fn(&i32), clobber: fn()) {
    // CHECK-LABEL: fn indirect_ref_1
    // CHECK: out => [[out:_.*]];
    // CHECK: [[y1_0:_.*]] = no_retag copy ((*_1).0: &i32);
    // CHECK: [[reborrow_y1_0:_.*]] = &(*[[y1_0]]);
    // CHECK: copy [[out]](move [[reborrow_y1_0]]) -> [
    // CHECK: [[y2_0:_.*]] = no_retag copy ((*_1).0: &i32);
    // CHECK: [[reborrow_y2_0:_.*]] = &(*[[y2_0]]);
    // CHECK: copy [[out]](move [[reborrow_y2_0]]) -> [
    out((*x).0);
    clobber();
    out((*x).0);
}

// EMIT_MIR indirect_ref.indirect_ref_2.GVN.diff
#[inline(never)]
pub fn indirect_ref_2(x: &(&i32,), out: fn(&i32), clobber: fn()) {
    // CHECK-LABEL: fn indirect_ref_2
    // CHECK: out => [[out:_.*]];
    // CHECK: [[y1:_.*]] = copy (*_1);
    // CHECK: [[y1_0:_.*]] = no_retag copy ([[y1]].0: &i32);
    // CHECK: [[reborrow_y1_0:_.*]] = &(*[[y1_0]]);
    // CHECK: copy [[out]](move [[reborrow_y1_0]]) -> [
    // CHECK: [[y2:_.*]] = copy (*_1);
    // CHECK: [[y2_0:_.*]] = no_retag copy ([[y2]].0: &i32);
    // CHECK: [[reborrow_y2_0:_.*]] = &(*[[y2_0]]);
    // CHECK: copy [[out]](move [[reborrow_y2_0]]) -> [
    let y1 = *x;
    out((y1).0);
    clobber();
    let y2 = *x;
    out((y2).0);
}

// EMIT_MIR indirect_ref.indirect_ref_t_1.GVN.diff
#[inline(never)]
pub fn indirect_ref_t_1<T: Copy + Freeze>(x: &(T,), out: fn(T), clobber: fn()) {
    // CHECK-LABEL: fn indirect_ref_t_1
    // CHECK: out => [[out:_.*]];
    // CHECK: [[y1_0:_.*]] = copy ((*_1).0: T);
    // CHECK: copy [[out]](move [[y1_0]]) -> [
    // CHECK: [[y2_0:_.*]] = copy ((*_1).0: T);
    // CHECK: copy [[out]](move [[y2_0]]) -> [
    out((*x).0);
    clobber();
    out((*x).0);
}

// EMIT_MIR indirect_ref.indirect_ref_t_2.GVN.diff
#[inline(never)]
pub fn indirect_ref_t_2<T: Copy + Freeze>(x: &(T,), out: fn(T), clobber: fn()) {
    // CHECK-LABEL: fn indirect_ref_t_2
    // CHECK: out => [[out:_.*]];
    // CHECK: [[y1:_.*]] = copy (*_1);
    // CHECK: [[y1_0:_.*]] = copy ([[y1]].0: T);
    // CHECK: copy [[out]](move [[y1_0]]) -> [
    // CHECK: [[y2:_.*]] = copy (*_1);
    // CHECK: [[y2_0:_.*]] = copy ([[y2]].0: T);
    // CHECK: copy [[out]](move [[y2_0]]) -> [
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
    // CHECK-LABEL: fn indirect_adt
    // CHECK: out => [[out:_.*]];
    // CHECK: [[y1:_.*]] = copy (*_1);
    // CHECK: copy [[out]](copy [[y1]]) -> [
    // CHECK: [[y2:_.*]] = copy (*_1);
    // CHECK: copy [[out]](copy [[y2]]) -> [
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
