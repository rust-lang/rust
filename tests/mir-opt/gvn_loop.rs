//@ test-mir-pass: GVN
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes

#![crate_type = "lib"]
#![feature(core_intrinsics, rustc_attrs)]

pub enum Value {
    V0(i32),
    V1,
}

// Check that we do not use the dereferenced value of `val_alias` when returning.

// EMIT_MIR gvn_loop.loop_deref_mut.GVN.diff
fn loop_deref_mut(val: &mut Value) -> Value {
    // CHECK-LABEL: fn loop_deref_mut(
    // CHECK: [[VAL_REF:_.*]] = get::<Value>(
    // CHECK: [[V:_.*]] = copy (((*[[VAL_REF]]) as V0).0: i32);
    // CEHCK-NOT: copy (*[[VAL_REF]]);
    // CHECK: [[RET:_*]] = Value::V0(copy [[V]]);
    // CEHCK-NOT: copy (*[[VAL_REF]]);
    // CHECK: _0 = move [[RET]]
    let val_alias: &Value = get(val);
    let mut stop = false;
    let Value::V0(v) = *val_alias else { unsafe { core::intrinsics::unreachable() } };
    loop {
        let v = Value::V0(v);
        if stop {
            return v;
        }
        stop = true;
        *val = Value::V1;
    }
}

#[inline(never)]
#[rustc_nounwind]
fn get<T>(v: &T) -> &T {
    v
}
