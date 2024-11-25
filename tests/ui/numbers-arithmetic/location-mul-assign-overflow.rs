//@ run-fail
//@ check-run-results:location-mul-assign-overflow.rs

fn main() {
    let mut a: u8 = 255;
    a *= &2;
}
