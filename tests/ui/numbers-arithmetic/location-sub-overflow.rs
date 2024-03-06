//@ run-fail
//@ ignore-wasm32
//@ error-pattern:location-sub-overflow.rs

fn main() {
    let _: u8 = 0 - &1;
}
