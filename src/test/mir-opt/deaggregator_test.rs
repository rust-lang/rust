struct Baz {
    x: usize,
    y: f32,
    z: bool,
}

fn bar(a: usize) -> Baz {
    Baz { x: a, y: 0.0, z: false }
}

fn main() {
    // Make sure the function actually gets instantiated.
    bar(0);
}

// END RUST SOURCE
// START rustc.bar.Deaggregator.before.mir
// bb0: {
//     ...
//     _2 = _1;
//     ...
//     _0 = Baz { x: move _2, y: const 0f32, z: const false };
//     ...
//     return;
// }
// END rustc.bar.Deaggregator.before.mir
// START rustc.bar.Deaggregator.after.mir
// bb0: {
//     ...
//     _2 = _1;
//     ...
//     (_0.0: usize) = move _2;
//     (_0.1: f32) = const 0f32;
//     (_0.2: bool) = const false;
//     ...
//     return;
// }
// END rustc.bar.Deaggregator.after.mir
