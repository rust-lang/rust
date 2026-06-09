//@ run-fail
//@ error-pattern:location-sub-assign-overflow.rs

fn main() {
    let mut a: u8 = 0;
    a -= &1;
}
