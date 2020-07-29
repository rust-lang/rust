// Test that the initial version of Rust coverage injects count_code_region() placeholder calls,
// at the top of each function. The placeholders are later converted into LLVM instrprof.increment
// intrinsics, during codegen.

// needs-profiler-support
// compile-flags: -Zinstrument-coverage
// EMIT_MIR instrument_coverage.main.InstrumentCoverage.diff
// EMIT_MIR instrument_coverage.bar.InstrumentCoverage.diff
fn main() {
    loop {
        if bar() {
            break;
        }
    }
}

#[inline(never)]
fn bar() -> bool {
    true
}
