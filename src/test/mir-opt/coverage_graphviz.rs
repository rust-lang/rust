// Test that `-C instrument-coverage` with `-Z dump-mir-graphviz` generates a graphviz (.dot file)
// rendering of the `BasicCoverageBlock` coverage control flow graph, with counters and
// expressions.

// needs-profiler-support
// compile-flags: -C instrument-coverage -Z dump-mir-graphviz
// EMIT_MIR coverage_graphviz.main.InstrumentCoverage.0.dot
// EMIT_MIR coverage_graphviz.bar.InstrumentCoverage.0.dot
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
