//@ edition:2018
//@ aux-crate:external=ambiguous-trait-reexport.rs

mod m1 {
    pub trait Trait {
        fn method1(&self) {}
    }
    impl Trait for u8 {}
}
mod m2 {
    pub trait Trait {
        fn method2(&self) {}
    }
    impl Trait for u8 {}
}
mod m1_reexport {
    pub use crate::m1::Trait;
}
mod m2_reexport {
    pub use crate::m2::Trait;
}

mod ambig_reexport {
    pub use crate::m1::*;
    pub use crate::m2::*;
}

fn test1() {
    // Create an ambiguous import for `Trait` in one order
    use m1::*;
    use m2::*;
    0u8.method1(); //~ WARNING Use of ambiguously glob imported trait `Trait` [ambiguous_glob_imported_traits]
                   //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    0u8.method2(); //~ ERROR: no method named `method2` found for type `u8` in the current scope
}

fn test2() {
    // Create an ambiguous import for `Trait` in another order
    use m2::*;
    use m1::*;
    0u8.method1(); //~ ERROR: no method named `method1` found for type `u8` in the current scope
    0u8.method2(); //~ WARNING Use of ambiguously glob imported trait `Trait` [ambiguous_glob_imported_traits]
                   //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn test_indirect_reexport() {
    use m1_reexport::*;
    use m2_reexport::*;
    0u8.method1(); //~ WARNING Use of ambiguously glob imported trait `Trait` [ambiguous_glob_imported_traits]
                   //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    0u8.method2(); //~ ERROR: no method named `method2` found for type `u8` in the current scope
}

fn test_ambig_reexport() {
    use ambig_reexport::*;
    0u8.method1(); //~ WARNING Use of ambiguously glob imported trait `Trait` [ambiguous_glob_imported_traits]
                   //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    0u8.method2(); //~ ERROR: no method named `method2` found for type `u8` in the current scope
}

fn test_external() {
    use external::m1::*;
    use external::m2::*;
    0u8.method1(); //~ WARNING Use of ambiguously glob imported trait `Trait` [ambiguous_glob_imported_traits]
                   //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    0u8.method2(); //~ ERROR: no method named `method2` found for type `u8` in the current scope
}

fn test_external_indirect_reexport() {
    use external::m1_reexport::*;
    use external::m2_reexport::*;
    0u8.method1(); //~ WARNING Use of ambiguously glob imported trait `Trait` [ambiguous_glob_imported_traits]
                   //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    0u8.method2(); //~ ERROR: no method named `method2` found for type `u8` in the current scope
}

fn test_external_ambig_reexport() {
    use external::ambig_reexport::*;
    0u8.method1(); //~ WARNING Use of ambiguously glob imported trait `Trait` [ambiguous_glob_imported_traits]
                   //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    0u8.method2(); //~ ERROR: no method named `method2` found for type `u8` in the current scope
}

fn main() {}
