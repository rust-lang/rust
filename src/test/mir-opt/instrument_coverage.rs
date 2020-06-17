// Test that the initial version of Rust coverage injects count_code_region() placeholder calls,
// at the top of each function. The placeholders are later converted into LLVM instrprof.increment
// intrinsics, during codegen.

// needs-profiler-support
// compile-flags: -Zinstrument-coverage
// EMIT_MIR rustc.main.InstrumentCoverage.diff
// EMIT_MIR rustc.bar.InstrumentCoverage.diff
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
