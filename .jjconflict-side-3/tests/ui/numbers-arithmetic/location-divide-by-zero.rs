//@ run-fail
//@ error-pattern:location-divide-by-zero.rs

// https://github.com/rust-lang/rust/issues/114814

fn main() {
    let _ = 1 / &0;
}
