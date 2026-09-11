//! See [`S`].

// Check that this isn't an ICE

// https://github.com/rust-lang/rust/issues/100241

mod foo {
    pub use inner::S;
    //~^ ERROR unresolved import `inner`
}

use foo::*;
use foo::S;
