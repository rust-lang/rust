//@ run-pass

fn foo<const N: usize, const K: usize>(_data: [u32; N]) -> [u32; K] {
    [0; K]
}
fn main() {
    let _a = foo::<_, 2>([0, 1, 2]);
}
