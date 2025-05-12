//! See [`S`].

// Check that this isn't an ICE
//@ should-fail

// https://github.com/rust-lang/rust/issues/100241

mod foo {
    pub use inner::S;
    //~^ ERROR unresolved imports `inner`, `foo::S`
}

use foo::*;
use foo::S;
