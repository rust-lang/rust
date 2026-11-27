// Test for a regression introduced by splitting module scope into two scopes
// (similar to issue #145575).

//@ check-pass
//@ edition: 2018..

#[macro_use]
mod one {
    // Macro that is in a different module, but still in scope due to `macro_use`
    macro_rules! mac { () => {} }
    pub(crate) use mac;
}

mod other {
    macro_rules! mac { () => {} }
    pub(crate) use mac;
}

// Single import of the same in the current module.
use one::mac;
// Glob import of a different macro in the current module (should be an ambiguity).
use other::*;

fn main() {
    mac!(); // OK for now, the ambiguity is not reported
}
