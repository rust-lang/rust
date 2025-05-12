// Check that changes to environment variables are propagated to `option_env!`.
//
// This test is intentionally written to not use any `#[cfg(rpass*)]`, to
// _really_ test that we re-compile if the environment variable changes.

//@ revisions: rpass1 rpass2 rpass3 rpass4
//@ [rpass1]unset-rustc-env:EXAMPLE_ENV
//@ [rpass1]unset-exec-env:EXAMPLE_ENV
//@ [rpass2]rustc-env:EXAMPLE_ENV=one
//@ [rpass2]exec-env:EXAMPLE_ENV=one
//@ [rpass3]rustc-env:EXAMPLE_ENV=two
//@ [rpass3]exec-env:EXAMPLE_ENV=two
//@ [rpass4]unset-rustc-env:EXAMPLE_ENV
//@ [rpass4]unset-exec-env:EXAMPLE_ENV

fn main() {
    assert_eq!(option_env!("EXAMPLE_ENV"), std::env::var("EXAMPLE_ENV").ok().as_deref());
}
