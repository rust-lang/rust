//! The `simplify_aggregate_to_copy` mir-opt introduced in
//! <https://github.com/rust-lang/rust/pull/128299> caused a miscompile because the initial
//! implementation
//!
//! > introduce[d] new dereferences without checking for aliasing
//!
//! This test demonstrates the behavior, and should be adjusted or removed when fixing and relanding
//! the mir-opt.
#![crate_type = "lib"]
//@ test-mir-pass: GVN
#![allow(internal_features)]
#![feature(core_intrinsics, custom_mir, rustc_attrs)]

use std::intrinsics::mir::*;

// EMIT_MIR simplify_aggregate_to_copy_miscompile.foo.GVN.diff
fn foo(v: &mut Option<i32>) -> Option<i32> {
    // CHECK-LABEL: fn foo(
    // CHECK-SAME: [[v:_.*]]: &mut Option<i32>
    // CHECK: [[v_alias_1:_.*]] = &(*_1)
    // CHECK-NEXT: [[v_alias_2:_.*]] = get::<Option<i32>>(move [[v_alias_1]])
    // CHECK: (*[[v]]) = const Option::<i32>::None;
    // CHECK-NOT: _0 = copy (*[[v_alias_2]])
    // CHECK: _0 = Option::<i32>::Some
    // CHECK-NOT: _0 = copy (*[[v_alias_2]])
    if let &Some(col) = get(v) {
        *v = None;
        return Some(col);
    } else {
        unsafe { std::intrinsics::unreachable() }
    }
}

pub enum Value {
    V0(i32),
    V1(i32),
}

// EMIT_MIR simplify_aggregate_to_copy_miscompile.set_discriminant.GVN.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
fn set_discriminant(v: &mut Value) -> Value {
    // CHECK-LABEL: fn set_discriminant(
    mir! {
        let v_: &Value;
        {
            Call(v_ = get(v), ReturnTo(ret), UnwindUnreachable())
        }
        ret = {
            let col: i32 = Field(Variant(*v_, 0), 0);
            SetDiscriminant(*v, 1);
            RET = Value::V0(col);
            Return()
        }
    }
}

#[inline(never)]
#[rustc_nounwind]
fn get<T>(v: &T) -> &T {
    v
}
