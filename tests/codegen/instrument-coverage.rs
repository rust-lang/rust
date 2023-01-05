// Test that `-Cinstrument-coverage` creates expected __llvm_profile_filename symbol in LLVM IR.

// needs-profiler-support
// compile-flags: -Cinstrument-coverage

// CHECK: @__llvm_profile_filename = {{.*}}"default_%m_%p.profraw\00"{{.*}}

#![crate_type="lib"]

#[inline(never)]
fn some_function() {

}

pub fn some_other_function() {
    some_function();
}
