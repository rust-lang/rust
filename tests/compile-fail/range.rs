#![feature(step_by)]
#![feature(plugin)]
#![plugin(clippy)]

struct NotARange;
impl NotARange {
    fn step_by(&self, _: u32) {}
}

#[deny(range_step_by_zero, range_zip_with_len)]
fn main() {
    (0..1).step_by(0); //~ERROR Range::step_by(0) produces an infinite iterator
    // No warning for non-zero step
    (0..1).step_by(1);

    (1..).step_by(0); //~ERROR Range::step_by(0) produces an infinite iterator

    let x = 0..1;
    x.step_by(0); //~ERROR Range::step_by(0) produces an infinite iterator

    // No error, not a range.
    let y = NotARange;
    y.step_by(0);

    let _v1 = vec![1,2,3];
    let _v2 = vec![4,5];
    let _x = _v1.iter().zip(0.._v1.len()); //~ERROR It is more idiomatic to use _v1.iter().enumerate()
    let _y = _v1.iter().zip(0.._v2.len()); // No error
}
