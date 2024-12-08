//@ run-pass
//@ rustc-env:MY_VAR=tadam
//@ compile-flags: --env-set MY_VAR=123abc -Zunstable-options

// This test ensures that variables provided with `--env` take precedence over
// variables from environment.
fn main() {
    assert_eq!(env!("MY_VAR"), "123abc");
}
