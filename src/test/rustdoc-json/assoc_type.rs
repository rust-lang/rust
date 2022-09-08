// Regression test for <https://github.com/rust-lang/rust/issues/98547>.

// @has "$.index[*][?(@.name=='Trait')]"
// @has "$.index[*][?(@.name=='AssocType')]"
// @has "$.index[*][?(@.name=='S')]"
// @has "$.index[*][?(@.name=='S2')]"

pub trait Trait {
    type AssocType;
}

impl<T> Trait for T {
    type AssocType = Self;
}

pub struct S;

/// Not needed for the #98547 ICE to occur, but added to maximize the chance of
/// getting an ICE in the future. See
/// <https://github.com/rust-lang/rust/pull/98548#discussion_r908219164>
pub struct S2;
