//@ add-minicore
//@ compile-flags: --target bpfel-unknown-none -Copt-level=0
//@ needs-llvm-components: bpf
#![feature(no_core, linkage)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

unsafe extern "C" {
    #[unsafe(link_section = ".ksyms")]
    fn bpf_kfunc();
    #[unsafe(link_section = ".ksyms")]
    static bpf_ksym: u32;
    #[unsafe(link_section = ".ksyms")]
    #[linkage = "extern_weak"]
    static bpf_weak_ksym: *const u32;
    fn plain_fn();
}

// CHECK: @bpf_ksym = external
// CHECK-SAME: section ".ksyms"
// CHECK: @bpf_weak_ksym = extern_weak
// CHECK-SAME: section ".ksyms"

// CHECK-LABEL: define void @call_kfunc(
#[no_mangle]
pub fn call_kfunc() {
    unsafe { bpf_kfunc() }
}

#[no_mangle]
pub fn read_ksym() -> u32 {
    unsafe { bpf_ksym }
}

#[no_mangle]
pub fn weak_ksym() -> *const u32 {
    unsafe { bpf_weak_ksym }
}

#[no_mangle]
pub fn call_plain() {
    unsafe { plain_fn() }
}

// CHECK: declare void @bpf_kfunc() {{.*}} section ".ksyms"
// CHECK: declare void @plain_fn()
// CHECK-NOT: section
