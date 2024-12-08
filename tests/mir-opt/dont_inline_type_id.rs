// skip-filecheck
//@ test-mir-pass: Inline
//@ compile-flags: --crate-type=lib -C panic=abort

use std::any::{Any, TypeId};

struct A<T: ?Sized + 'static> {
    a: i32,
    b: T,
}

// EMIT_MIR dont_inline_type_id.call.Inline.diff
pub fn call<T: ?Sized + 'static>(s: &T) -> TypeId {
    s.type_id()
}
