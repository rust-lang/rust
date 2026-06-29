//@ compile-flags: -Zno-profiler-runtime -Zunstable-options
//@ compile-flags: -Cinstrument-coverage=presence-only -Copt-level=0

// Verify that presence-only coverage selects LLVM's byte-counter ABI and that
// the instrumentation pass lowers each probe to a one-byte covered store.

pub fn covered() -> bool {
    true
}

fn main() {
    covered();
}

// CHECK: @__llvm_profile_raw_version = {{.*}}constant i64 1152921504606846986

// CHECK: @[[COUNTER:__profc_.*single_byte7covered]] = {{private|internal}} global
// CHECK-SAME: [{{[0-9]+}} x i8] c"\FF
// CHECK-SAME: align 1

// CHECK: define{{.*}} @_R{{[a-zA-Z0-9_]+}}single_byte7covered
// CHECK-NOT: atomicrmw add
// CHECK: store i8 0, ptr @[[COUNTER]], align 1

// CHECK: declare void @llvm.instrprof.cover(ptr, i64, i32, i32)
// CHECK-NOT: declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
