// Test that __llvm_profile_counter_bias does not get internalized by lto.

//@ ignore-apple -runtime-counter-relocation not honored on Mach-O
//@ compile-flags: -Cprofile-generate -Cllvm-args=-runtime-counter-relocation -Clto=fat
//@ compile-flags: -Zno-profiler-runtime
//@ no-prefer-dynamic

// CHECK: @__llvm_profile_counter_bias = {{.*}}global

pub fn main() {}
