// Tests that clearing a `Vec` of a type without drop glue lowers to an
// unconditional store of the new length, without a comparison and branch
// guarding it.
// See <https://github.com/rust-lang/rust/issues/45459>.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @clear_vec(
// CHECK-NOT: icmp
// CHECK-NOT: br {{.*}}
// CHECK: store i{{[0-9]+}} 0
// CHECK-NOT: br {{.*}}
// CHECK: ret void
#[no_mangle]
pub fn clear_vec(v: &mut Vec<f32>) {
    v.clear();
}
