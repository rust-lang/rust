//@ run-fail
//@ error-pattern:location-add-assign-overflow.rs

fn main() {
    let mut a: u8 = 255;
    a += &1;
}
