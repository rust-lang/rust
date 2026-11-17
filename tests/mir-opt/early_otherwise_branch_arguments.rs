//@ test-mir-pass: EarlyOtherwiseBranch

#![feature(custom_mir, core_intrinsics)]
#![allow(dead_code, internal_features, unused_assignments)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

pub struct Wrapper {
    value: (u64,),
}

pub union U {
    value: u64,
    empty: (),
}

pub struct UnionWrapper {
    value: U,
}

#[repr(u8)]
pub enum E {
    A = 1,
    B = 2,
}

#[custom_mir(dialect = "runtime")]
pub fn unchanged_scalar_is_hoisted(q: u64, x: u64) -> u64 {
    // CHECK-LABEL: fn unchanged_scalar_is_hoisted(
    // CHECK: Ne(copy _1, copy _2)
    mir! {
        { match q { 1 => one, 2 => two, _ => fallback, } }
        one = { match x { 1 => ret1, _ => fallback, } }
        two = { match x { 2 => ret2, _ => fallback, } }
        ret1 = { RET = 100; Return() }
        ret2 = { RET = 200; Return() }
        fallback = { RET = 999; Return() }
    }
}

#[custom_mir(dialect = "runtime")]
pub fn unchanged_fields_are_hoisted(q: u64, x: Wrapper) -> u64 {
    // CHECK-LABEL: fn unchanged_fields_are_hoisted(
    // CHECK: Ne(copy _1, copy {{.*}}_2
    mir! {
        { match q { 1 => one, 2 => two, _ => fallback, } }
        one = { match x.value.0 { 1 => ret1, _ => fallback, } }
        two = { match x.value.0 { 2 => ret2, _ => fallback, } }
        ret1 = { RET = 100; Return() }
        ret2 = { RET = 200; Return() }
        fallback = { RET = 999; Return() }
    }
}

#[custom_mir(dialect = "runtime")]
pub fn uninitialized_local_is_not_hoisted(q: u64) -> u64 {
    // CHECK-LABEL: fn uninitialized_local_is_not_hoisted(
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy _1)
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy _2)
    // CHECK-NOT: Ne(
    mir! {
        let x: u64;
        { match q { 1 => init1, 2 => init2, _ => parent, } }
        init1 = { x = 1; Goto(parent) }
        init2 = { x = 2; Goto(parent) }
        parent = { match q { 1 => one, 2 => two, _ => fallback, } }
        one = { match x { 1 => ret1, _ => fallback, } }
        two = { match x { 2 => ret2, _ => fallback, } }
        ret1 = { RET = 100; Return() }
        ret2 = { RET = 200; Return() }
        fallback = { RET = 999; Return() }
    }
}

#[custom_mir(dialect = "runtime")]
pub unsafe fn union_field_is_not_hoisted(q: u64, x: UnionWrapper) -> u64 {
    // CHECK-LABEL: fn union_field_is_not_hoisted(
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy _1)
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy {{.*}}_2
    // CHECK-NOT: Ne(
    mir! {
        { match q { 1 => one, 2 => two, _ => fallback, } }
        one = { match x.value.value { 1 => ret1, _ => fallback, } }
        two = { match x.value.value { 2 => ret2, _ => fallback, } }
        ret1 = { RET = 100; Return() }
        ret2 = { RET = 200; Return() }
        fallback = { RET = 999; Return() }
    }
}

#[custom_mir(dialect = "runtime")]
pub fn borrowed_field_is_not_hoisted(q: u64, x: Wrapper) -> u64 {
    // CHECK-LABEL: fn borrowed_field_is_not_hoisted(
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy _1)
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy {{.*}}_2
    // CHECK-NOT: Ne(
    mir! {
        let p: *const u64;
        let unused: ();
        {
            p = &raw const x.value.0;
            Call(unused = opaque(p), ReturnTo(parent), UnwindUnreachable())
        }
        parent = { match q { 1 => one, 2 => two, _ => fallback, } }
        one = { match x.value.0 { 1 => ret1, _ => fallback, } }
        two = { match x.value.0 { 2 => ret2, _ => fallback, } }
        ret1 = { RET = 100; Return() }
        ret2 = { RET = 200; Return() }
        fallback = { RET = 999; Return() }
    }
}

#[custom_mir(dialect = "runtime")]
pub fn moved_field_is_not_hoisted(q: u64, x: Wrapper) -> u64 {
    // CHECK-LABEL: fn moved_field_is_not_hoisted(
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy _1)
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy {{.*}}_2
    // CHECK-NOT: Ne(
    mir! {
        let unused: ();
        { match q { 3 => consume, _ => parent, } }
        consume = {
            Call(unused = opaque(Move(x.value.0)), ReturnTo(parent), UnwindUnreachable())
        }
        parent = { match q { 1 => one, 2 => two, _ => fallback, } }
        one = { match x.value.0 { 1 => ret1, _ => fallback, } }
        two = { match x.value.0 { 2 => ret2, _ => fallback, } }
        ret1 = { RET = 100; Return() }
        ret2 = { RET = 200; Return() }
        fallback = { RET = 999; Return() }
    }
}

#[custom_mir(dialect = "runtime")]
pub fn reset_argument_is_not_hoisted(q: u64, x: u64) -> u64 {
    // CHECK-LABEL: fn reset_argument_is_not_hoisted(
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy _1)
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy _2)
    // CHECK-NOT: Ne(
    mir! {
        { match q { 3 => reset, _ => parent, } }
        reset = { StorageDead(x); StorageLive(x); Goto(parent) }
        parent = { match q { 1 => one, 2 => two, _ => fallback, } }
        one = { match x { 1 => ret1, _ => fallback, } }
        two = { match x { 2 => ret2, _ => fallback, } }
        ret1 = { RET = 100; StorageDead(x); Return() }
        ret2 = { RET = 200; StorageDead(x); Return() }
        fallback = { RET = 999; StorageDead(x); Return() }
    }
}

#[custom_mir(dialect = "runtime")]
pub fn written_argument_is_not_hoisted(q: u64, x: u64) -> u64 {
    // CHECK-LABEL: fn written_argument_is_not_hoisted(
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy _1)
    // CHECK-NOT: Ne(
    // CHECK: switchInt(copy _2)
    // CHECK-NOT: Ne(
    mir! {
        { x = 7; Goto(parent) }
        parent = { match q { 1 => one, 2 => two, _ => fallback, } }
        one = { match x { 1 => ret1, _ => fallback, } }
        two = { match x { 2 => ret2, _ => fallback, } }
        ret1 = { RET = 100; Return() }
        ret2 = { RET = 200; Return() }
        fallback = { RET = 999; Return() }
    }
}

#[custom_mir(dialect = "runtime")]
pub fn discriminant_is_not_hoisted(q: u8, x: E) -> u64 {
    // CHECK-LABEL: fn discriminant_is_not_hoisted(
    // CHECK: bb0: {
    // CHECK-NEXT: switchInt(copy _1)
    // CHECK: [[DISCR:_.*]] = discriminant(_2);
    // CHECK-NEXT: switchInt(move [[DISCR]])
    // CHECK: [[DISCR]] = discriminant(_2);
    // CHECK-NEXT: switchInt(move [[DISCR]])
    mir! {
        let discr: u8;
        { match q { 1 => one, 2 => two, _ => fallback, } }
        one = { discr = Discriminant(x); match Move(discr) { 1 => ret1, _ => fallback, } }
        two = { discr = Discriminant(x); match Move(discr) { 2 => ret2, _ => fallback, } }
        ret1 = { RET = 100; Return() }
        ret2 = { RET = 200; Return() }
        fallback = { RET = 999; Return() }
    }
}

// CHECK-LABEL: fn opaque
#[inline(never)]
fn opaque<T>(_: T) {}
