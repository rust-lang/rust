// Regression test for <https://github.com/rust-lang/rust/issues/98547>.

// @has assoc_type.json
// @has - "$.index[*][?(@.name=='Trait')]"
// @has - "$.index[*][?(@.name=='AssocType')]"
// @has - "$.index[*][?(@.name=='S')]"

pub trait Trait {
    type AssocType;
}

impl<T> Trait for T {
    type AssocType = Self;
}

pub struct S;
