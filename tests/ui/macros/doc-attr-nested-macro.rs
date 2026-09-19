//! Regression test for <https://github.com/rust-lang/rust/issues/44851>.
//! Test doc attr macro doesn't ICE with macro as an argument.
//@ check-pass

macro_rules! a {
    () => { "a" }
}

macro_rules! b {
    ($doc:expr) => {
        #[doc = $doc]
        pub struct B;
    }
}

b!(a!());

fn main() {}
