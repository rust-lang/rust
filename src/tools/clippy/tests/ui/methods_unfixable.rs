#![warn(clippy::filter_next)]
//@no-rustfix
fn main() {}

pub fn issue10029() {
    let iter = (0..10);
    let _ = iter.filter(|_| true).next();
    //~^ filter_next
}
