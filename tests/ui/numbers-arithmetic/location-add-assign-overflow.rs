//@ run-fail
//@ check-run-results:location-add-assign-overflow.rs

fn main() {
    let mut a: u8 = 255;
    a += &1;
}
