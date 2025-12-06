//@ edition:2018..

#![feature(decl_macro)]

mod m {
    // Glob import in macro namespace
    mod inner { pub macro mac() {} }
    pub use inner::*;

    // Macro-expanded single import in macro namespace
    macro_rules! define_mac {
        () => { pub macro mac() {} }
    }
    define_mac!();
}

// Ambiguous reexport
pub use m::*;
