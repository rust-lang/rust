// This test ensures that associated types don't crash rustdoc jump to def.

//@ compile-flags: -Zunstable-options --generate-link-to-definition


#![crate_name = "foo"]

//@ has 'src/foo/jump-to-def-ice-assoc-types.rs.html'

pub trait Trait {
    type Node;
}

pub fn y<G: Trait>() {
    struct X<G>(G);

    impl<G: Trait> Trait for X<G> {
        type Node = G::Node;
    }
}
