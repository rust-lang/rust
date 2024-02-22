//! See [`S`].

// Check that this isn't an ICE
//@ should-fail

mod foo {
    pub use inner::S;
    //~^ ERROR unresolved imports `inner`, `foo::S`
}

use foo::*;
use foo::S;
