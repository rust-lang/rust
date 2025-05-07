// Check that changes to environment variables are propagated to `env!`.
//
// This test is intentionally written to not use any `#[cfg(rpass*)]`, to
// _really_ test that we re-compile if the environment variable changes.

//@ revisions: cfail1 rpass2 rpass3 cfail4
//@ [cfail1]unset-rustc-env:EXAMPLE_ENV
//@ [rpass2]rustc-env:EXAMPLE_ENV=one
//@ [rpass2]exec-env:EXAMPLE_ENV=one
//@ [rpass3]rustc-env:EXAMPLE_ENV=two
//@ [rpass3]exec-env:EXAMPLE_ENV=two
//@ [cfail4]unset-rustc-env:EXAMPLE_ENV

fn main() {
    assert_eq!(env!("EXAMPLE_ENV"), std::env::var("EXAMPLE_ENV").unwrap());
    //[cfail1]~^ ERROR environment variable `EXAMPLE_ENV` not defined at compile time
    //[cfail4]~^^ ERROR environment variable `EXAMPLE_ENV` not defined at compile time
}
