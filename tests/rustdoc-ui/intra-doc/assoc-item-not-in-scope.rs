#![deny(rustdoc::broken_intra_doc_links)]

#[derive(Debug)]
/// Link to [`S::fmt`]
//~^ ERROR unresolved link
pub struct S;

pub mod inner {
    use std::fmt::Debug;
    use super::S;

    /// Link to [`S::fmt`]
    pub fn f() {}
}

pub mod ambiguous {
    use std::fmt::{Display, Debug};
    use super::S;

    /// Link to [`S::fmt`]
    pub fn f() {}
}
