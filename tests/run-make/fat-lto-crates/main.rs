// Crosses the crate boundary inside the fat core: calls, a shared static,
// and a trait object whose vtable lives in the other crate.

use hot::{Add, Op, TABLE};

#[inline(never)]
fn mix(ops: &[Box<dyn Op>], seed: u32) -> u32 {
    ops.iter().fold(seed, |acc, op| op.apply(acc))
}

fn main() {
    let ops: Vec<Box<dyn Op>> = vec![Box::new(Add(17)), Box::new(Add(3))];
    let v = tail::finish(hot::twist(mix(&ops, 41)).wrapping_add(TABLE[2]));
    println!("{v}");
}
