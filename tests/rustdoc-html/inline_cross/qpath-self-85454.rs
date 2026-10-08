//@ aux-build:issue-85454.rs
//@ build-aux-docs
#![crate_name = "foo"]
// https://github.com/rust-lang/rust/issues/85454

extern crate issue_85454;

//@ has foo/trait.TryFromBreak.html
//@ has - '//pre[@class="rust item-decl"]' 'pub trait TryFromBreak<R = <Self as Try>::Break> { // Required method fn from_break(residual: R) -> Self; }'
pub trait TryFromBreak<R = <Self as Try>::Break> {
    fn from_break(residual: R) -> Self;
}

pub trait Try: TryFromBreak {
    type Output;
    type Break;
    fn from_output(output: Self::Output) -> Self;
    fn branch(self) -> ControlFlow<Self::Break, Self::Output>;
}

pub enum ControlFlow<B, C = ()> {
    Continue(C),
    Break(B),
}

pub mod reexport {
    //@ has foo/reexport/trait.TryFromBreak.html
    //@ has - '//pre[@class="rust item-decl"]' 'pub trait TryFromBreak<R = <Self as Try>::Break> { // Required method fn from_break(residual: R) -> Self; }'
    pub use issue_85454::*;
}
