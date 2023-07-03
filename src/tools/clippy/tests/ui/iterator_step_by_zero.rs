#![allow(clippy::useless_vec)]
#[warn(clippy::iterator_step_by_zero)]
fn main() {
    let _ = vec!["A", "B", "B"].iter().step_by(0);
    let _ = "XXX".chars().step_by(0);
    let _ = (0..1).step_by(0);

    // No error, not an iterator.
    let y = NotIterator;
    y.step_by(0);

    // No warning for non-zero step
    let _ = (0..1).step_by(1);

    let _ = (1..).step_by(0);
    let _ = (1..=2).step_by(0);

    let x = 0..1;
    let _ = x.step_by(0);

    // check const eval
    let v1 = vec![1, 2, 3];
    let _ = v1.iter().step_by(2 / 3);
}

struct NotIterator;
impl NotIterator {
    fn step_by(&self, _: u32) {}
}
