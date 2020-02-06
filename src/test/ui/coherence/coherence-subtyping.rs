// Test that two distinct impls which match subtypes of one another
// yield coherence errors (or not) depending on the variance.
//
// Note: This scenario is currently accepted, but as part of the
// universe transition (#56105) may eventually become an error.

// revisions: old re
// check-pass

trait TheTrait {
    fn foo(&self) {}
}

impl TheTrait for for<'a, 'b> fn(&'a u8, &'b u8) -> &'a u8 {}

impl TheTrait for for<'a> fn(&'a u8, &'a u8) -> &'a u8 {
    //[re]~^ WARNING conflicting implementation
    //[re]~^^ WARNING this was previously accepted by the compiler but is being phased out
    //[old]~^^^ WARNING conflicting implementation
    //[old]~^^^^ WARNING this was previously accepted by the compiler but is being phased out
}

fn main() {}
