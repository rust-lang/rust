// skip-filecheck
//@ test-mir-pass: GVN

#![crate_type = "lib"]
#![feature(core_intrinsics, rustc_attrs)]

pub enum Value {
    V0(i32),
    V1,
}

// EMIT_MIR gvn_loop.loop_deref_mut.GVN.diff
fn loop_deref_mut(val: &mut Value) -> Value {
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
