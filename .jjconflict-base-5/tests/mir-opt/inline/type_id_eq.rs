#![feature(core_intrinsics)]
//@ test-mir-pass: Inline
//@ compile-flags: --crate-type=lib -C panic=abort

use std::any::{Any, TypeId};
use std::intrinsics::type_id_eq;

struct A<T: ?Sized + 'static> {
    a: i32,
    b: T,
}

// EMIT_MIR type_id_eq.call.Inline.diff
// CHECK-LABEL: fn call(
pub fn call(a: TypeId, b: TypeId) -> bool {
    // CHECK: as u128 (Transmute)
    // CHECK: as u128 (Transmute)
    // CHECK: Eq
    type_id_eq(a, b)
}
