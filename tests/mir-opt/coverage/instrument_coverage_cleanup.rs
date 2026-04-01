// Test that CleanupPostBorrowck cleans up the marker statements that are
// inserted during MIR building (after InstrumentCoverage is done with them),
// but leaves the statements that were added by InstrumentCoverage.
//
// Removed statement kinds: BlockMarker, SpanMarker
// Retained statement kinds: VirtualCounter

//@ test-mir-pass: InstrumentCoverage
//@ compile-flags: -Cinstrument-coverage -Zcoverage-options=branch -Zno-profiler-runtime

// EMIT_MIR instrument_coverage_cleanup.main.InstrumentCoverage.diff
// EMIT_MIR instrument_coverage_cleanup.main.CleanupPostBorrowck.diff
fn main() {
    if !core::hint::black_box(true) {}
}

// CHECK-NOT: Coverage::BlockMarker
// CHECK-NOT: Coverage::SpanMarker
// CHECK:     Coverage::VirtualCounter
// CHECK-NOT: Coverage::BlockMarker
// CHECK-NOT: Coverage::SpanMarker
