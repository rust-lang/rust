// run-pass
// Test that two distinct impls which match subtypes of one another
// yield coherence errors (or not) depending on the variance.

trait Contravariant {
    fn foo(&self) { }
}

impl Contravariant for for<'a,'b> fn(&'a u8, &'b u8) -> &'a u8 {
}

impl Contravariant for for<'a> fn(&'a u8, &'a u8) -> &'a u8 {
}

///////////////////////////////////////////////////////////////////////////

trait Covariant {
    fn foo(&self) { }
}

impl Covariant for for<'a,'b> fn(&'a u8, &'b u8) -> &'a u8 {
}

impl Covariant for for<'a> fn(&'a u8, &'a u8) -> &'a u8 {
}

///////////////////////////////////////////////////////////////////////////

trait Invariant {
    fn foo(&self) { }
}

impl Invariant for for<'a,'b> fn(&'a u8, &'b u8) -> &'a u8 {
}

impl Invariant for for<'a> fn(&'a u8, &'a u8) -> &'a u8 {
}

fn main() { }
