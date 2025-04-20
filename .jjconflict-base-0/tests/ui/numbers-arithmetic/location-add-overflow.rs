//@ run-fail
//@ error-pattern:location-add-overflow.rs

fn main() {
    let _: u8 = 255 + &1;
}
