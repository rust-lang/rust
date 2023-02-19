// aux-build:issue-85454.rs
// build-aux-docs
#![crate_name = "foo"]

extern crate issue_85454;

// @has foo/trait.FromResidual.html
// @has - '//pre[@class="rust item-decl"]' 'pub trait FromResidual<R = <Self as Try>::Residual> { // Required method fn from_residual(residual: R) -> Self; }'
pub trait FromResidual<R = <Self as Try>::Residual> {
    fn from_residual(residual: R) -> Self;
}

pub trait Try: FromResidual {
    type Output;
    type Residual;
    fn from_output(output: Self::Output) -> Self;
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
}

pub enum ControlFlow<B, C = ()> {
    Continue(C),
    Break(B),
}

pub mod reexport {
    // @has foo/reexport/trait.FromResidual.html
    // @has - '//pre[@class="rust item-decl"]' 'pub trait FromResidual<R = <Self as Try>::Residual> { // Required method fn from_residual(residual: R) -> Self; }'
    pub use issue_85454::*;
}
