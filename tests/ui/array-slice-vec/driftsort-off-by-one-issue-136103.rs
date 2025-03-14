//@ run-pass
// Ensures that driftsort doesn't crash under specific slice
// length and memory size.
// Based on the example given in https://github.com/rust-lang/rust/issues/136103.
fn main() {
    let n = 127;
    let mut objs: Vec<_> =
        (0..n).map(|i| [(i % 2) as u8; 125001]).collect();
    objs.sort();
}
