//@ test-mir-pass: GVN
//@ compile-flags: -C opt-level=2

#![feature(core_intrinsics)]

fn generic<T>() {}

// Since `type_id` contains provenance, we cannot turn it into an integer,
// but for the purposes of this test, it is sufficient to use the length of the
// type name as the first 8 bytes of the `u128` and the first 8 bytes of the type name
// as the rest of the bytes of the `u128`. The main thing being tested is the result of
// calling this function being simple enough that comparisons of it are turned into
// a direct MIR `Eq` bin op.
const fn type_id_of_val<T: 'static>(_: &T) -> u128 {
    let name = std::intrinsics::type_name::<T>();
    let len = name.len() as u64;
    let len = u64::to_be_bytes(len);
    let mut ret = [0; 16];
    let mut i = 0;
    while i < 8 {
        ret[i] = len[i];
        i += 1;
    }
    while i < 16 {
        ret[i] = name.as_bytes()[i - 8];
        i += 1;
    }
    u128::from_be_bytes(ret)
}

// EMIT_MIR gvn_type_id_polymorphic.cursed_is_i32.GVN.diff
fn cursed_is_i32<T: 'static>() -> bool {
    // CHECK-LABEL: fn cursed_is_i32(
    // CHECK: _0 = Eq(const cursed_is_i32::<T>::{constant#0}, const cursed_is_i32::<T>::{constant#1});
    // CHECK-NEXT: return;
    (const { type_id_of_val(&generic::<T>) } == const { type_id_of_val(&generic::<i32>) })
}

fn main() {
    dbg!(cursed_is_i32::<i32>());
}
