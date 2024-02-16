//@ run-pass
//@ rustc-env:MY_ENV=/
// Ensures that variables not defined through `--env-set` are still available.

fn main() {
    assert!(!env!("MY_ENV").is_empty());
}
