//@ run-fail
//@ error-pattern:location-mul-overflow.rs

fn main() {
    let _: u8 = 255 * &2;
}
