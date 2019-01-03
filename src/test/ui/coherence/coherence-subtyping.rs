// Test that two distinct impls which match subtypes of one another
// yield coherence errors (or not) depending on the variance.

trait TheTrait {
    fn foo(&self) { }
}

impl TheTrait for for<'a,'b> fn(&'a u8, &'b u8) -> &'a u8 {
}

impl TheTrait for for<'a> fn(&'a u8, &'a u8) -> &'a u8 {
    //~^ ERROR
}

fn main() { }
