//@ compile-flags: -Copt-level=3
//@ needs-deterministic-layouts

// Currently Vec<T> and &[T] have layouts that start with (pointer, len)
// which makes the conversion branchless.
// A nice-to-have property, not guaranteed.
#![crate_type = "cdylib"]

// CHECK-LABEL: @branchless_cow_slices
#[no_mangle]
pub fn branchless_cow_slices<'a>(cow: &'a std::borrow::Cow<'a, [u8]>) -> &'a [u8] {
    // CHECK-NOT: br
    // CHECK-NOT: select
    // CHECK-NOT: icmp
    // CHECK: ret { ptr, {{i32|i64}} }
    &*cow
}
