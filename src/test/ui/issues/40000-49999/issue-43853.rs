// run-pass
// ignore-wasm32-bare compiled with panic=abort by default

use std::panic;

fn test() {
    wait(|| panic!());
}

fn wait<T, F: FnOnce() -> T>(f: F) -> F::Output {
    From::from(f())
}

fn main() {
    let result = panic::catch_unwind(move || test());
    assert!(result.is_err());
}
