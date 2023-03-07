// Regression test for <https://github.com/rust-lang/rust/issues/97986>.

// @has "$.index[*][?(@.name=='f')]"
// @has "$.index[*][?(@.name=='AssocTy')]"
// @has "$.index[*][?(@.name=='AssocConst')]"

pub mod m {
    pub struct S;
}

pub trait F {
    type AssocTy;
    const AssocConst: usize;
    fn f() -> m::S;
}

impl<T> F for T {
    type AssocTy = u32;
    const AssocConst: usize = 0;
    fn f() -> m::S {
        m::S
    }
}
