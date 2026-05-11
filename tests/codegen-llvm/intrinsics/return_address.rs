//@ ignore-wasm

#![crate_type = "lib"]
#![feature(core_intrinsics, return_address)]

// CHECK-LABEL: @call_return_address_intrinsic
#[no_mangle]
#[inline(never)]
pub fn call_return_address_intrinsic() -> *const () {
    // CHECK: call ptr @llvm.returnaddress(i32 0)
    core::intrinsics::return_address()
}
