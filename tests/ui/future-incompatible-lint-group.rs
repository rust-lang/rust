// Ensure that the future_incompatible lint group only includes
// lints for changes that are not tied to an edition
#![deny(future_incompatible)]

trait Tr {
    // Warn only since this is not a `future_incompatible` lint
    fn f(u8) {} //~ WARN anonymous parameters are deprecated
                //~| WARN this is accepted in the current edition
}

pub mod submodule {
    // Error since this is a `future_incompatible` lint
    #![doc(test(some_test))]
        //~^ ERROR this attribute can only be applied at the crate level
        //~| WARN this was previously accepted by the compiler
}

fn main() {}
