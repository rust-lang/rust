//@ run-pass
//@ rustc-env:MY_ENV=/

fn main() {
    assert!(!env!("MY_ENV").is_empty());
}
