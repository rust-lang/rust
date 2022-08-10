#![crate_name = "foo"]
#![feature(generic_associated_types)]

pub trait Trait {
    type Gat<'a>;
}

// Make sure that the elided lifetime shows up

// @has foo/type.T.html
// @hastext - "pub type T = "
// @hastext - "&lt;'_&gt;"
pub type T = fn(&<() as Trait>::Gat<'_>);
