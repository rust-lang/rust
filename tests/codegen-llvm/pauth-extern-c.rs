// ignore-tidy-linelength
//@ only-aarch64-unknown-linux-pauthtest
//@ add-minicore

//@ revisions: O0_PAUTH O3_PAUTH O0_PAUTH-ELF-GOT O3_PAUTH-ELF-GOT O0_NO_PAUTH O3_NO_PAUTH

//@ [O0_PAUTH] needs-llvm-components: aarch64
//@ [O0_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0
//@ [O3_PAUTH] needs-llvm-components: aarch64
//@ [O3_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=3
//@ [O0_PAUTH-ELF-GOT] needs-llvm-components: aarch64
//@ [O0_PAUTH-ELF-GOT] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0 -Z ptrauth-elf-got
//@ [O3_PAUTH-ELF-GOT] needs-llvm-components: aarch64
//@ [O3_PAUTH-ELF-GOT] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=3 -Z ptrauth-elf-got
//@ [O0_NO_PAUTH] needs-llvm-components: aarch64
//@ [O0_NO_PAUTH] compile-flags: --target=aarch64-unknown-linux-gnu -C opt-level=0
//@ [O3_NO_PAUTH] needs-llvm-components: aarch64
//@ [O3_NO_PAUTH] compile-flags: --target=aarch64-unknown-linux-gnu -C opt-level=3

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core)]

extern crate minicore;

type FnPtr = unsafe extern "C" fn(i32, i32) -> i32;
// O0_NO_PAUTH-NOT: "ptrauth"(i32
// O3_NO_PAUTH-NOT: "ptrauth"(i32

// O0_PAUTH: define {{.*}}test_entry
// O3_PAUTH: define {{.*}}test_entry
#[no_mangle]
pub unsafe extern "C" fn test_entry(x: usize) {
    // O0_PAUTH: call{{.*}}_RNvCshUtaFcP1mZ5_14pauth_extern_c7call_it(ptr ptrauth (ptr @external_c_callee, i32 0), i32 5, i32 7)
    // O3_PAUTH: call{{.*}}_RNvCshUtaFcP1mZ5_14pauth_extern_c7call_it(ptr{{.*}}ptrauth (ptr @external_c_callee, i32 0), i32{{.*}}5, i32{{.*}}7)
    let _ = call_it(external_c_callee, 5, 7);
}

// O0_PAUTH: define {{.*}}pauth_extern_c7call_it{{.*}} #[[ATTR_O0_1:[0-9]+]]
// O3_PAUTH: define {{.*}}pauth_extern_c7call_it{{.*}} #[[ATTR_O3_1:[0-9]+]]
#[inline(never)]
pub fn call_it(fn_ptr: FnPtr, arg_1: i32, arg_2: i32) -> i32 {
    // O0_PAUTH: call i32 %fn_ptr(i32 %arg_1, i32 %arg_2){{.*}}[ "ptrauth"(i32 0, i64 0) ]
    // O3_PAUTH: call{{.*}}i32 %fn_ptr(i32{{.*}}%arg_1, i32{{.*}}%arg_2){{.*}}[ "ptrauth"(i32 0, i64 0) ]
    unsafe { fn_ptr(arg_1, arg_2) }
}

extern "C" {
    fn external_c_callee(a: i32, b: i32) -> i32;
}

// O0_PAUTH-CHECK: attributes #[[ATTR_O0_1]] = { {{.*}}"aarch64-jump-table-hardening"
// O0_PAUTH-CHECK-SAME: "ptrauth-auth-traps"
// O0_PAUTH-CHECK-SAME: "ptrauth-calls"
// O0_PAUTH-CHECK-SAME: "ptrauth-indirect-gotos"
// O0_PAUTH-CHECK-SAME: "ptrauth-returns"

// O3_PAUTH-CHECK: attributes #[[ATTR_O3_1]] = { {{.*}}"aarch64-jump-table-hardening"
// O3_PAUTH-CHECK-SAME: "ptrauth-auth-traps"
// O3_PAUTH-CHECK-SAME: "ptrauth-calls"
// O3_PAUTH-CHECK-SAME: "ptrauth-indirect-gotos"
// O3_PAUTH-CHECK-SAME: "ptrauth-returns"

// O0_PAUTH-ELF-GOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-elf-got", i32 1}
// O0_PAUTH-NOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-elf-got", i32 1}
// O0_PAUTH: !{{[0-9]+}} = !{i32 1, !"ptrauth-sign-personality", i32 1}
// O3_PAUTH-ELF-GOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-elf-got", i32 1}
// O3_PAUTH-NOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-elf-got", i32 1}
// O3_PAUTH: !{{[0-9]+}} = !{i32 1, !"ptrauth-sign-personality", i32 1}

// O0_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-elf-got", i32 1}
// O0_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-sign-personality", i32 1}
// O3_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-elf-got", i32 1}
// O3_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-sign-personality", i32 1}
