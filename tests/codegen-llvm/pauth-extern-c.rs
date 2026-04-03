// ignore-tidy-linelength
//@ only-aarch64-unknown-linux-pauthtest
//@ revisions: O0_PAUTH O3_PAUTH O0_PAUTH-ELF-GOT O3_PAUTH-ELF-GOT O0_NO_PAUTH O3_NO_PAUTH

//@ [O0_PAUTH] needs-llvm-components: aarch64
//@ [O0_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0
//@ [O3_PAUTH] needs-llvm-components: aarch64
//@ [O3_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=3
//@ [O0_PAUTH-ELF-GOT] needs-llvm-components: aarch64
//@ [O0_PAUTH-ELF-GOT] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0 -Z pauth_enable_elf_got
//@ [O3_PAUTH-ELF-GOT] needs-llvm-components: aarch64
//@ [O3_PAUTH-ELF-GOT] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=3 -Z pauth_enable_elf_got
//@ [O0_NO_PAUTH] needs-llvm-components: aarch64
//@ [O0_NO_PAUTH] compile-flags: --target=aarch64-unknown-linux-gnu -C opt-level=0
//@ [O3_NO_PAUTH] needs-llvm-components: aarch64
//@ [O3_NO_PAUTH] compile-flags: --target=aarch64-unknown-linux-gnu -C opt-level=3

use std::hint::black_box;

type FnPtr = unsafe extern "C" fn(i32, i32) -> i32;

// O0_NO_PAUTH-NOT: "ptrauth"(i32
// O3_NO_PAUTH-NOT: "ptrauth"(i32

// O0_PAUTH: define {{.*}}pauth_extern_c4main
// O3_PAUTH: define {{.*}}pauth_extern_c4main
fn main() {
    // O0_PAUTH: ptr ptrauth (ptr @add_from_c, i32 0)
    // O3_PAUTH: ptr ptrauth (ptr @add_from_c, i32 0)
    let add_ptr: FnPtr = black_box(add_from_c);
    // O0_PAUTH: call i32 @{{.*}}pauth_extern_c7call_it{{.*}}(ptr {{.*}})
    let _sum = call_it(add_ptr, 5, 7);
    assert!(12 == _sum);
}

// O0_PAUTH: define {{.*}}pauth_extern_c7call_it{{.*}} #[[ATTR_O0_1:[0-9]+]]
// O3_PAUTH: define {{.*}}pauth_extern_c7call_it{{.*}} #[[ATTR_O3_1:[0-9]+]]
#[inline(never)]
fn call_it(fn_ptr: FnPtr, arg_1: i32, arg_2: i32) -> i32 {
    // O0_PAUTH: call i32 %fn_ptr(i32 %arg_1, i32 %arg_2) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    // O3_PAUTH: call {{.*}} i32 %fn_ptr(i32 {{.*}}, i32 {{.*}}) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    unsafe { fn_ptr(arg_1, arg_2) }
}

extern "C" {
    fn add_from_c(a: i32, b: i32) -> i32;
}

// O0_PAUTH-CHECK: attributes #[[ATTR_O0_1]] = { {{.*}}"aarch64-jump-table-hardening"
// O0_PAUTH-CHECK-DAG: "ptrauth-auth-traps"
// O0_PAUTH-CHECK-DAG: "ptrauth-calls"
// O0_PAUTH-CHECK-DAG: "ptrauth-indirect-gotos"
// O0_PAUTH-CHECK-DAG: "ptrauth-returns"

// O3_PAUTH-CHECK: attributes #[[ATTR_O3_1]] = { {{.*}}"aarch64-jump-table-hardening"
// O3_PAUTH-CHECK-DAG: "ptrauth-auth-traps"
// O3_PAUTH-CHECK-DAG: "ptrauth-calls"
// O3_PAUTH-CHECK-DAG: "ptrauth-indirect-gotos"
// O3_PAUTH-CHECK-DAG: "ptrauth-returns"

// O0_PAUTH-ELF-GOT: !{{[0-9]+}} = !{i32 7, !"ptrauth-elf-got", i32 1}
// O0_PAUTH-NOT: !{{[0-9]+}} = !{i32 7, !"ptrauth-elf-got", i32 1}
// O0_PAUTH: !{{[0-9]+}} = !{i32 7, !"ptrauth-sign-personality", i32 1}
// O3_PAUTH-ELF-GOT: !{{[0-9]+}} = !{i32 7, !"ptrauth-elf-got", i32 1}
// O3_PAUTH-NOT: !{{[0-9]+}} = !{i32 7, !"ptrauth-elf-got", i32 1}
// O3_PAUTH: !{{[0-9]+}} = !{i32 7, !"ptrauth-sign-personality", i32 1}

// O0_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 7, !"ptrauth-elf-got", i32 1}
// O0_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 7, !"ptrauth-sign-personality", i32 1}
// O3_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 7, !"ptrauth-elf-got", i32 1}
// O3_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 7, !"ptrauth-sign-personality", i32 1}
