//! Verify that the wasm `externref` lang type lowers to `ptr addrspace(10)`
//! in function signatures and stays a direct SSA value.

//@ add-minicore
//@ compile-flags: -Copt-level=3 --target wasm32-unknown-unknown
//@ needs-llvm-components: webassembly

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;

#[lang = "externref"]
#[non_exhaustive]
pub struct externref;

extern "C" {
    fn create_ref() -> externref;
    fn use_ref(v: externref);
}

// CHECK: define {{.*}}ptr addrspace(10) @describe(ptr addrspace(10) {{.*}}%v)
#[no_mangle]
pub extern "C" fn describe(v: externref) -> externref {
    v
}

// CHECK-LABEL: @roundtrip
#[no_mangle]
pub extern "C" fn roundtrip() {
    // CHECK: %[[V:.+]] = {{.*}}call {{.*}}ptr addrspace(10) @create_ref()
    // CHECK: call void @use_ref(ptr addrspace(10) {{.*}}%[[V]])
    unsafe { use_ref(create_ref()) }
}
