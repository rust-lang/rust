//@ check-pass
//@ edition: 2021

// Regression test for <https://github.com/rust-lang/rust/issues/164158>.
// A nested use group with no path in front of it made rustdoc panic.

use {{}};

pub mod inner {
    pub struct S;
    pub struct T;
}

pub use {inner::S, {inner::T}};
