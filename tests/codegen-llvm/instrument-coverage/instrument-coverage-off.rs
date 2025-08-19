// Test that `-Cinstrument-coverage=off` does not add coverage instrumentation to LLVM IR.

//@ compile-flags: -Zno-profiler-runtime
//@ revisions: n no off false_ zero
//@ [n] compile-flags: -Cinstrument-coverage=n
//@ [no] compile-flags: -Cinstrument-coverage=no
//@ [off] compile-flags: -Cinstrument-coverage=off
//@ [false_] compile-flags: -Cinstrument-coverage=false
//@ [zero] compile-flags: -Cinstrument-coverage=0

// CHECK-NOT: __llvm_profile_filename
// CHECK-NOT: __llvm_coverage_mapping

#![crate_type = "lib"]

#[inline(never)]
fn some_function() {}

pub fn some_other_function() {
    some_function();
}
