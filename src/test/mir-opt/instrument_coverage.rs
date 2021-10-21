// Test that `-C instrument-coverage` injects Coverage statements. The Coverage Counter statements
// are later converted into LLVM instrprof.increment intrinsics, during codegen.

// needs-profiler-support
// ignore-windows
// compile-flags: -C instrument-coverage --remap-path-prefix={{src-base}}=/the/src

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

// Note that the MIR with injected coverage intrinsics includes references to source locations,
// including the source file absolute path. Typically, MIR pretty print output with file
// references are safe because the file prefixes are substituted with `$DIR`, but in this case
// the file references are encoded as function arguments, with an `Operand` type representation
// (`Slice` `Allocation` interned byte array) that cannot be normalized by simple substitution.
//
// The first workaround is to use the `SourceMap`-supported `--remap-path-prefix` option; however,
// the implementation of the `--remap-path-prefix` option currently joins the new prefix and the
// remaining source path with an OS-specific path separator (`\` on Windows). This difference still
// shows up in the byte array representation of the path, causing Windows tests to fail to match
// blessed results baselined with a `/` path separator.
//
// Since this `mir-opt` test does not have any significant platform dependencies, other than the
// path separator differences, the final workaround is to disable testing on Windows.
