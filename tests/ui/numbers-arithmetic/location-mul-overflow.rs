//@ run-fail
//@ ignore-wasm32
//@ error-pattern:location-mul-overflow.rs

fn main() {
    let _: u8 = 255 * &2;
}
