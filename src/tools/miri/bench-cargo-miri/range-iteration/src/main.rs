//! This generates a lot of work for the AllocId part of the GC.
fn main() {
    // The end of the range is just chosen to make the benchmark run for a few seconds.
    for _ in 0..50_000 {}
}
