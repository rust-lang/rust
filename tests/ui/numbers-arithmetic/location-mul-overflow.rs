//@ run-fail
//@ check-run-results:location-mul-overflow.rs

fn main() {
    let _: u8 = 255 * &2;
}
