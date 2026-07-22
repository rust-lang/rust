//! Regression test for <https://github.com/rust-lang/rust/issues/32782>
macro_rules! bar (
    () => ()
);

macro_rules! foo (
    () => (
        #[allow_internal_unstable()]
        //~^ ERROR the `allow_internal_unstable` attribute side-steps
        //~| ERROR `#[allow_internal_unstable]` attribute cannot be used on macro calls
        bar!();
    );
);

foo!();
fn main() {}
