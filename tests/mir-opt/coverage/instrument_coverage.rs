// Test that `-C instrument-coverage` injects Coverage statements.
// The Coverage::CounterIncrement statements are later converted into LLVM
// instrprof.increment intrinsics, during codegen.

//@ test-mir-pass: InstrumentCoverage
//@ compile-flags: -Cinstrument-coverage -Zno-profiler-runtime

// EMIT_MIR instrument_coverage.main.InstrumentCoverage.diff
// CHECK-LABEL: fn main()
// CHECK: coverage Code { bcb: bcb{{[0-9]+}} } =>
// CHECK: bb0:
// CHECK: Coverage::VirtualCounter
fn main() {
    loop {
        if bar() {
            break;
        }
    }
}

// EMIT_MIR instrument_coverage.bar.InstrumentCoverage.diff
// CHECK-LABEL: fn bar()
// CHECK: coverage Code { bcb: bcb{{[0-9]+}} } =>
// CHECK: bb0:
// CHECK: Coverage::VirtualCounter
#[inline(never)]
fn bar() -> bool {
    true
}
