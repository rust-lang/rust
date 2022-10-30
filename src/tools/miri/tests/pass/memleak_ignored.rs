//@compile-flags: -Zmiri-ignore-leaks

fn main() {
    std::mem::forget(Box::new(42));
}
