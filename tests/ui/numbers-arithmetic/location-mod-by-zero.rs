//@ run-fail
//@ ignore-wasm32
//@ error-pattern:location-mod-by-zero.rs

fn main() {
    let _ = 1 % &0;
}
