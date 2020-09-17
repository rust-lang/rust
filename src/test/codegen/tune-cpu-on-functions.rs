// This test makes sure that functions get annotated with the proper
// "tune-cpu" attribute in LLVM.

// no-prefer-dynamic
// ignore-tidy-linelength
// compile-flags: -C no-prepopulate-passes -C panic=abort -C linker-plugin-lto -Cpasses=name-anon-globals -Z tune-cpu=generic

#![crate_type = "staticlib"]

// CHECK-LABEL: define {{.*}} @exported() {{.*}} #0
#[no_mangle]
pub extern fn exported() {
    not_exported();
}

// CHECK-LABEL: ; tune_cpu_on_functions::not_exported
// CHECK-NEXT: ; Function Attrs:
// CHECK-NEXT: define {{.*}}() {{.*}} #0
fn not_exported() {}

// CHECK: attributes #0 = {{.*}} "tune-cpu"="{{.*}}"
