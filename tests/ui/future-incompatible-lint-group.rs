// Ensure that the future_incompatible lint group only includes
// lints for changes that are not tied to an edition
#![deny(future_incompatible)]

// Error since this is a `future_incompatible` lint
macro_rules! m { ($i) => {} } //~ ERROR missing fragment specifier
                              //~| WARN this was previously accepted

trait Tr {
    // Warn only since this is not a `future_incompatible` lint
    fn f(u8) {} //~ WARN anonymous parameters are deprecated
                //~| WARN this is accepted in the current edition
}

fn main() {}
