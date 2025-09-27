//@ compile-flags: -C panic=abort
//@ revisions: NONWASM WASM WASMEXN
//@ [NONWASM] ignore-wasm32
//@ [WASM] only-wasm32
//@ [WASMEXN] only-wasm32
//@ [WASMEXN] compile-flags: -Ctarget-feature=+exception-handling

#![crate_type = "lib"]

extern "C-unwind" {
    fn bar();
}

// CHECK: Function Attrs:{{.*}}nounwind
// CHECK-NEXT: define{{.*}}void @foo
// Handle both legacy and v0 symbol mangling.
// NONWASM: call void @{{.*core9panicking19panic_cannot_unwind}}
// WASMEXN: call void @{{.*core9panicking19panic_cannot_unwind}}
// WASM-NOT: call void @{{.*core9panicking19panic_cannot_unwind}}
#[no_mangle]
pub unsafe extern "C" fn foo() {
    bar();
}
