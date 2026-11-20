//! Regression test for <https://github.com/rust-lang/rust/issues/146398>.

// The panic happens while the JSON emitter fills in its `rendered` field, which is the
// path `cargo` takes, so this has to be checked with the default JSON error format.
//@ compile-flags: --diagnostic-width=30
// ignore-tidy-file-tab

fn main() {
    let _: &[u8] = [0,												0];
    //~^ ERROR mismatched types
}
