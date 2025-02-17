//@ test-mir-pass: GVN
//@ compile-flags: -C opt-level=2

#![feature(core_intrinsics)]

fn generic<T>() {}

const fn type_id_of_val<T: 'static>(_: &T) -> u128 {
    std::intrinsics::type_id::<T>()
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
