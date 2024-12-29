//@ run-fail
//@ check-run-results:location-add-overflow.rs

fn main() {
    let _: u8 = 255 + &1;
}
