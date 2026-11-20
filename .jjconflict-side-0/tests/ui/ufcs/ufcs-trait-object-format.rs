//@ run-pass
//! Regression test for <https://github.com/rust-lang/rust/issues/20676>.
//! Error was that we didn't support
//! UFCS-style calls to a method in `Trait` where `Self` was bound to a
//! trait object of type `Trait`.
//! See also <https://github.com/rust-lang/rust/blob/ec2cc76/tests/ui/traits/ufcs-object.rs>.

use std::fmt;

fn main() {
    let a: &dyn fmt::Debug = &1;
    let _ = format!("{:?}", a);
}
