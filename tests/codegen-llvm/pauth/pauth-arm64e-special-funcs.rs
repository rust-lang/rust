//@ only-arm64e-apple-darwin
//@ compile-flags: -Copt-level=0

// CHECK: define {{.*}} @__rust_try{{.*}} [[ATTR:#[0-9]+]]
// CHECK: invoke void %{{.*}}({{.*}}) [ "ptrauth"(i32 0, i64 0) ]
// CHECK: call void %{{.*}}({{.*}}) [ "ptrauth"(i32 0, i64 0) ]
// CHECK: define {{.*}} @main{{.*}} [[ATTR]]
// CHECK: call i64 @{{.*}}(ptr ptrauth (ptr @{{.*}}, i32 0),
// CHECK: attributes [[ATTR]] = { {{.*}}"aarch64-jump-table-hardening"
// CHECK-SAME: "ptrauth-auth-traps"
// CHECK-SAME: "ptrauth-calls"
// CHECK-SAME: "ptrauth-indirect-gotos"
// CHECK-SAME: "ptrauth-returns"
// CHECK-NOT: !"ptrauth-sign-personality"
fn main() {
    let _ = std::panic::catch_unwind(|| panic!("BOOM"));
}
