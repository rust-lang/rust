#![crate_name = "foo"]

pub trait Trait {
    type Gat<'a>;
}

// Make sure that the elided lifetime shows up

// @has foo/type.T.html
// @hasraw - "pub type T = "
// @hasraw - "&lt;'_&gt;"
pub type T = fn(&<() as Trait>::Gat<'_>);
