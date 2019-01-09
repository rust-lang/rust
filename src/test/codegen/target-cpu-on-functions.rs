// This test makes sure that functions get annotated with the proper
// "target-cpu" attribute in LLVM.

// no-prefer-dynamic
// ignore-tidy-linelength
// compile-flags: -C no-prepopulate-passes -C panic=abort -Z cross-lang-lto -Cpasses=name-anon-globals

#![crate_type = "staticlib"]

// CHECK-LABEL: define {{.*}} @exported() {{.*}} #0
#[no_mangle]
pub extern fn exported() {
    not_exported();
}

// CHECK-LABEL: define {{.*}} @_ZN23target_cpu_on_functions12not_exported{{.*}}() {{.*}} #0
fn not_exported() {}

// CHECK: attributes #0 = {{.*}} "target-cpu"="{{.*}}"
