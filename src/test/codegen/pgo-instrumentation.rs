// Test that `-Zpgo-gen` creates expected instrumentation artifacts in LLVM IR.

// needs-profiler-support
// compile-flags: -Z pgo-gen -Ccodegen-units=1

// CHECK: @__llvm_profile_raw_version =
// CHECK: @__profc_{{.*}}pgo_instrumentation{{.*}}some_function{{.*}} = private global
// CHECK: @__profd_{{.*}}pgo_instrumentation{{.*}}some_function{{.*}} = private global
// CHECK: @__profc_{{.*}}pgo_instrumentation{{.*}}main{{.*}} = private global
// CHECK: @__profd_{{.*}}pgo_instrumentation{{.*}}main{{.*}} = private global
// CHECK: @__llvm_profile_filename = {{.*}}"default_%m.profraw\00"{{.*}}

#[inline(never)]
fn some_function() {

}

fn main() {
    some_function();
}
