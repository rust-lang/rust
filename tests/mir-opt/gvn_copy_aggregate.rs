//@ test-mir-pass: GVN
//@ compile-flags: -Cpanic=abort

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

struct AllCopy {
    a: i32,
    b: u64,
    c: [i8; 3],
}

// EMIT_MIR gvn_copy_aggregate.all_copy.GVN.diff
fn all_copy(v: &AllCopy) -> AllCopy {
    // CHECK-LABEL: fn all_copy(
    // CHECK: bb0: {
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK: _0 = copy (*_1);
    let a = v.a;
    let b = v.b;
    let c = v.c;
    AllCopy { a, b, c }
}

// EMIT_MIR gvn_copy_aggregate.all_copy_2.GVN.diff
fn all_copy_2(v: &&AllCopy) -> AllCopy {
    // CHECK-LABEL: fn all_copy_2(
    // CHECK: bb0: {
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK: [[V1:_.*]] = copy (*_1);
    // CHECK: _0 = copy (*[[V1]]);
    let a = v.a;
    let b = v.b;
    let c = v.c;
    AllCopy { a, b, c }
}

// EMIT_MIR gvn_copy_aggregate.all_copy_move.GVN.diff
fn all_copy_move(v: AllCopy) -> AllCopy {
    // CHECK-LABEL: fn all_copy_move(
    // CHECK: bb0: {
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK: _0 = copy _1;
    let a = v.a;
    let b = v.b;
    let c = v.c;
    AllCopy { a, b, c }
}

// EMIT_MIR gvn_copy_aggregate.all_copy_ret_2.GVN.diff
fn all_copy_ret_2(v: &AllCopy) -> (AllCopy, AllCopy) {
    // CHECK-LABEL: fn all_copy_ret_2(
    // CHECK: bb0: {
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK: [[V1:_.*]] = copy (*_1);
    // CHECK: [[V2:_.*]] = copy [[V1]];
    // CHECK: _0 = (copy [[V1]], copy [[V1]]);
    let a = v.a;
    let b = v.b;
    let c = v.c;
    (AllCopy { a, b, c }, AllCopy { a, b, c })
}

struct AllCopy2 {
    a: i32,
    b: u64,
    c: [i8; 3],
}

// EMIT_MIR gvn_copy_aggregate.all_copy_different_type.GVN.diff
fn all_copy_different_type(v: &AllCopy) -> AllCopy2 {
    // CHECK-LABEL: fn all_copy_different_type(
    // CHECK: bb0: {
    // CHECK: _0 = AllCopy2 { {{.*}} };
    let a = v.a;
    let b = v.b;
    let c = v.c;
    AllCopy2 { a, b, c }
}

struct SameType {
    a: i32,
    b: i32,
}

// EMIT_MIR gvn_copy_aggregate.same_type_different_index.GVN.diff
fn same_type_different_index(v: &SameType) -> SameType {
    // CHECK-LABEL: fn same_type_different_index(
    // CHECK: bb0: {
    // CHECK: _0 = SameType { {{.*}} };
    let a = v.b;
    let b = v.a;
    SameType { a, b }
}

// EMIT_MIR gvn_copy_aggregate.all_copy_has_changed.GVN.diff
fn all_copy_has_changed(v: &mut AllCopy) -> AllCopy {
    // CHECK-LABEL: fn all_copy_has_changed(
    // CHECK: bb0: {
    // CHECK: _0 = AllCopy { {{.*}} };
    let a = v.a;
    let b = v.b;
    let c = v.c;
    v.a = 1;
    AllCopy { a, b, c }
}

// FIXME: This can be simplified to `Copy`.
// EMIT_MIR gvn_copy_aggregate.all_copy_use_changed.GVN.diff
fn all_copy_use_changed(v: &mut AllCopy) -> AllCopy {
    // CHECK-LABEL: fn all_copy_use_changed(
    // CHECK: bb0: {
    // CHECK-NOT: _0 = copy (*_1);
    // CHECK: = AllCopy { {{.*}} };
    let mut a = v.a;
    v.a = 1;
    a = v.a;
    let b = v.b;
    let c = v.c;
    AllCopy { a, b, c }
}

// FIXME: This can be simplified to `Copy`.
// EMIT_MIR gvn_copy_aggregate.all_copy_use_changed_2.GVN.diff
fn all_copy_use_changed_2(v: &mut AllCopy) -> AllCopy {
    // CHECK-LABEL: fn all_copy_use_changed_2(
    // CHECK: bb0: {
    // CHECK-NOT: _0 = (*_1);
    // CHECK: = AllCopy { {{.*}} };
    let mut a = v.a;
    let b = v.b;
    let c = v.c;
    v.a = 1;
    a = v.a;
    AllCopy { a, b, c }
}

struct NestCopy {
    d: i32,
    all_copy: AllCopy,
}

// EMIT_MIR gvn_copy_aggregate.nest_copy.GVN.diff
fn nest_copy(v: &NestCopy) -> NestCopy {
    // CHECK-LABEL: fn nest_copy(
    // CHECK: bb0: {
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK-NOT: = NestCopy { {{.*}} };
    let a = v.all_copy.a;
    let b = v.all_copy.b;
    let c = v.all_copy.c;
    let all_copy = AllCopy { a, b, c };
    let d = v.d;
    NestCopy { d, all_copy }
}

enum Enum1 {
    A(AllCopy),
    B(AllCopy),
}

// EMIT_MIR gvn_copy_aggregate.enum_identical_variant.GVN.diff
fn enum_identical_variant(v: &Enum1) -> Enum1 {
    // CHECK-LABEL: fn enum_identical_variant(
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK: _0 = copy (*_1);
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK: _0 = copy (*_1);
    match v {
        Enum1::A(v) => {
            let a = v.a;
            let b = v.b;
            let c = v.c;
            let all_copy = AllCopy { a, b, c };
            Enum1::A(all_copy)
        }
        Enum1::B(v) => {
            let a = v.a;
            let b = v.b;
            let c = v.c;
            let all_copy = AllCopy { a, b, c };
            Enum1::B(all_copy)
        }
    }
}

// EMIT_MIR gvn_copy_aggregate.enum_different_variant.GVN.diff
fn enum_different_variant(v: &Enum1) -> Enum1 {
    // CHECK-LABEL: fn enum_different_variant(
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK: [[V1:_.*]] = copy (((*_1) as [[VARIANT1:.*]]).0: AllCopy);
    // CHECK: _0 = Enum1::[[VARIANT2:.*]](copy [[V1]]);
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK: [[V2:_.*]] = copy (((*_1) as [[VARIANT2]]).0: AllCopy);
    // CHECK: _0 = Enum1::[[VARIANT1]](copy [[V2]]);
    match v {
        Enum1::A(v) => {
            let a = v.a;
            let b = v.b;
            let c = v.c;
            let all_copy = AllCopy { a, b, c };
            Enum1::B(all_copy)
        }
        Enum1::B(v) => {
            let a = v.a;
            let b = v.b;
            let c = v.c;
            let all_copy = AllCopy { a, b, c };
            Enum1::A(all_copy)
        }
    }
}

enum AlwaysSome<T> {
    Some(T),
}

// Ensure that we do not access this local after `StorageDead`.
// EMIT_MIR gvn_copy_aggregate.remove_storage_dead.GVN.diff
fn remove_storage_dead<T>(f: fn() -> AlwaysSome<T>) -> AlwaysSome<T> {
    // CHECK-LABEL: fn remove_storage_dead(
    // CHECK: [[V1:_.*]] = copy _1() -> [return: [[BB1:bb.*]],
    // CHECK: [[BB1]]: {
    // CHECK-NOT: StorageDead([[V1]]);
    // CHECK: _0 = copy [[V1]];
    let v = {
        match f() {
            AlwaysSome::Some(v) => v,
        }
    };
    AlwaysSome::Some(v)
}

// EMIT_MIR gvn_copy_aggregate.remove_storage_dead_from_index.GVN.diff
#[custom_mir(dialect = "analysis")]
fn remove_storage_dead_from_index(f: fn() -> usize, v: [SameType; 5]) -> SameType {
    // CHECK-LABEL: fn remove_storage_dead_from_index(
    // CHECK: [[V1:_.*]] = copy _1() -> [return: [[BB1:bb.*]],
    // CHECK: [[BB1]]: {
    // CHECK-NOT: StorageDead([[V1]]);
    // CHECK-NOT: = SameType { {{.*}} };
    // CHECK: _0 = copy _2[[[V1]]];
    mir! {
        let index: usize;
        let a: i32;
        let b: i32;
        {
            StorageLive(index);
            Call(index = f(), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            a = v[index].a;
            b = v[index].b;
            StorageDead(index);
            RET = SameType { a, b };
            Return()
        }
    }
}
