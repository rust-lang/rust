// Test that `-Cprofile-generate` creates expected instrumentation artifacts in LLVM IR.
// Compiling with `-Cpanic=abort` because PGO+unwinding isn't supported on all platforms.

// needs-profiler-support
// compile-flags: -Cprofile-generate -Ccodegen-units=1 -Cpanic=abort

// CHECK: @__llvm_profile_raw_version =
// CHECK: @__profc_{{.*}}pgo_instrumentation{{.*}}some_function{{.*}} = private global
// CHECK: @__profd_{{.*}}pgo_instrumentation{{.*}}some_function{{.*}} = private global
// CHECK: @__profc_{{.*}}pgo_instrumentation{{.*}}some_other_function{{.*}} = private global
// CHECK: @__profd_{{.*}}pgo_instrumentation{{.*}}some_other_function{{.*}} = private global
// CHECK: @__llvm_profile_filename = {{.*}}"default_%m.profraw\00"{{.*}}

#![crate_type="lib"]

#[inline(never)]
fn some_function() {

}

pub fn some_other_function() {
    some_function();
}
