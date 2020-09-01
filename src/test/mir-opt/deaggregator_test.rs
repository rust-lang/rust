struct Baz {
    x: usize,
    y: f32,
    z: bool,
}

// EMIT_MIR deaggregator_test.bar.Deaggregator.diff
fn bar(a: usize) -> Baz {
    Baz { x: a, y: 0.0, z: false }
}

fn main() {
    // Make sure the function actually gets instantiated.
    bar(0);
}
