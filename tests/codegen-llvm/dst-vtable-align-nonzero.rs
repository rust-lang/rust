//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// This test checks that we annotate alignment loads from vtables with nonzero range metadata.
// The companion LLVM-21-only fold check lives in `dst-vtable-align-1-elide-check.rs`.

pub trait Trait {
    fn f(&self);
}

pub struct WrapperWithAlign2<T: ?Sized> {
    x: u16,
    y: T,
}

pub struct Struct<W: ?Sized> {
    _field: i8,
    dst: W,
}

// CHECK-LABEL: @does_not_eliminate_runtime_check_when_align_2
#[no_mangle]
pub fn does_not_eliminate_runtime_check_when_align_2(
    x: &Struct<WrapperWithAlign2<dyn Trait>>,
) -> &WrapperWithAlign2<dyn Trait> {
    // CHECK: {{icmp|llvm.umax}}
    // CHECK: ret
    &x.dst
}

// CHECK-LABEL: @align_load_from_align_of_val
#[no_mangle]
pub fn align_load_from_align_of_val(x: &dyn Trait) -> usize {
    // CHECK: {{%[0-9]+}} = load [[USIZE:i[0-9]+]], {{.+}} !range [[RANGE_META:![0-9]+]]
    core::mem::align_of_val(x)
}

// CHECK-LABEL: @align_load_from_vtable_align_intrinsic
#[no_mangle]
pub unsafe fn align_load_from_vtable_align_intrinsic(x: &dyn Trait) -> usize {
    let (data, vtable): (*const (), *const ()) = core::mem::transmute(x);
    // CHECK: {{%[0-9]+}} = load [[USIZE]], {{.+}} !range [[RANGE_META]]
    core::intrinsics::vtable_align(vtable)
}

// CHECK: [[RANGE_META]] = !{[[USIZE]] 1, [[USIZE]] [[#0x20000001]]
