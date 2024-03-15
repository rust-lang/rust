// Test that CleanupPostBorrowck cleans up the marker statements that are
// inserted during MIR building (after InstrumentCoverage is done with them),
// but leaves the statements that were added by InstrumentCoverage.
//
// Removed statement kinds: BlockMarker, SpanMarker
// Retained statement kinds: CounterIncrement, ExpressionUsed

//@ unit-test: InstrumentCoverage
//@ compile-flags: -Cinstrument-coverage -Zcoverage-options=branch -Zno-profiler-runtime
//@ compile-flags: --remap-path-prefix={{src-base}}=/the/src

// EMIT_MIR instrument_coverage_cleanup.main.InstrumentCoverage.diff
// EMIT_MIR instrument_coverage_cleanup.main.CleanupPostBorrowck.diff
fn main() {
    if !core::hint::black_box(true) {}
}

// CHECK-NOT: Coverage::BlockMarker
// CHECK-NOT: Coverage::SpanMarker
// CHECK:     Coverage::CounterIncrement
// CHECK-NOT: Coverage::BlockMarker
// CHECK-NOT: Coverage::SpanMarker
