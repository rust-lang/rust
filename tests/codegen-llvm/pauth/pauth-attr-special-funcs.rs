//@ only-pauthtest
//@ compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0
// Make sure that compiler generated functions (main wrapper and __rust_try) also have ptrauth
// attributes set correctly. Rustc only generates __rust_try at O0, so use that opt level for the
// test.

//@ needs-llvm-components: aarch64

use std::panic;

// Make sure that `rust_eh_personality` is not signed.
// CHECK: define internal i32 @{{.*}}lang_start{{.*}}pauth_attr_special_funcs(ptr %{{.*}}) unnamed_addr #[[#]] personality ptr @rust_eh_personality

// CHECK: define {{.*}} @__rust_try{{.*}} [[ATTR_TRY:#[0-9]+]] personality ptr @rust_eh_personality {
// CHECK: define {{.*}} @main{{.*}} [[ATTR_MAIN:#[0-9]+]]

// CHECK: attributes [[ATTR_TRY]] = { {{.*}}"aarch64-jump-table-hardening"
// CHECK-DAG: "ptrauth-auth-traps"
// CHECK-DAG: "ptrauth-calls"
// CHECK-DAG: "ptrauth-indirect-gotos"
// CHECK-DAG: "ptrauth-returns"

// CHECK: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// CHECK-SAME: "ptrauth-auth-traps"
// CHECK-SAME: "ptrauth-calls"
// CHECK-SAME: "ptrauth-indirect-gotos"
// CHECK-SAME: "ptrauth-returns"
fn main() {
    let _ = panic::catch_unwind(|| {
        panic!("BOOM");
    });
}

// CHECK: !{{[0-9]+}} = !{i32 1, !"ptrauth-sign-personality", i32 1}
