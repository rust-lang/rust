#![allow(clippy::useless_vec)]
#[warn(clippy::iterator_step_by_zero)]
fn main() {
    let _ = vec!["A", "B", "B"].iter().step_by(0);
    //~^ ERROR: `Iterator::step_by(0)` will panic at runtime
    //~| NOTE: `-D clippy::iterator-step-by-zero` implied by `-D warnings`
    let _ = "XXX".chars().step_by(0);
    //~^ ERROR: `Iterator::step_by(0)` will panic at runtime
    let _ = (0..1).step_by(0);
    //~^ ERROR: `Iterator::step_by(0)` will panic at runtime

    // No error, not an iterator.
    let y = NotIterator;
    y.step_by(0);

    // No warning for non-zero step
    let _ = (0..1).step_by(1);

    let _ = (1..).step_by(0);
    //~^ ERROR: `Iterator::step_by(0)` will panic at runtime
    let _ = (1..=2).step_by(0);
    //~^ ERROR: `Iterator::step_by(0)` will panic at runtime

    let x = 0..1;
    let _ = x.step_by(0);
    //~^ ERROR: `Iterator::step_by(0)` will panic at runtime

    // check const eval
    let v1 = vec![1, 2, 3];
    let _ = v1.iter().step_by(2 / 3);
    //~^ ERROR: `Iterator::step_by(0)` will panic at runtime
}

struct NotIterator;
impl NotIterator {
    fn step_by(&self, _: u32) {}
}
