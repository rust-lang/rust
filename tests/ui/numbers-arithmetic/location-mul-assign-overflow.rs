//@ run-fail
//@ ignore-wasm32
//@ error-pattern:location-mul-assign-overflow.rs

fn main() {
    let mut a: u8 = 255;
    a *= &2;
}
