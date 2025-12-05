// Test that `-Cprofile-generate` creates expected instrumentation artifacts in LLVM IR.

//@ compile-flags: -Zno-profiler-runtime
//@ compile-flags: -Cprofile-generate -Ccodegen-units=1

// CHECK: @__llvm_profile_raw_version =
// CHECK-DAG: @__profc_{{.*}}pgo_instrumentation{{.*}}some_function{{.*}} = {{.*}}global
// CHECK-DAG: @__profd_{{.*}}pgo_instrumentation{{.*}}some_function{{.*}} = {{.*}}global
// CHECK-DAG: @__profc_{{.*}}pgo_instrumentation{{.*}}some_other_function{{.*}} = {{.*}}global
// CHECK-DAG: @__profd_{{.*}}pgo_instrumentation{{.*}}some_other_function{{.*}} = {{.*}}global
// CHECK: @__llvm_profile_filename = {{.*}}"default_%m.profraw\00"{{.*}}

#![crate_type = "lib"]

#[inline(never)]
fn some_function() {}

pub fn some_other_function() {
    some_function();
}
