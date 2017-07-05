#![feature(iterator_step_by)]
#![feature(inclusive_range_syntax)]
#![feature(plugin)]
#![plugin(clippy)]

struct NotARange;
impl NotARange {
    fn step_by(&self, _: u32) {}
}

#[warn(iterator_step_by_zero, range_zip_with_len)]
fn main() {
    let _ = (0..1).step_by(0);
    // No warning for non-zero step
    let _ = (0..1).step_by(1);

    let _ = (1..).step_by(0);
    let _ = (1...2).step_by(0);

    let x = 0..1;
    let _ = x.step_by(0);

    // No error, not a range.
    let y = NotARange;
    y.step_by(0);

    let v1 = vec![1,2,3];
    let v2 = vec![4,5];
    let _x = v1.iter().zip(0..v1.len());
    let _y = v1.iter().zip(0..v2.len()); // No error

    // check const eval
    let _ = v1.iter().step_by(2/3);
}
