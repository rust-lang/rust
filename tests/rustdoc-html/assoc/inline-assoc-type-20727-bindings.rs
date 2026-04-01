//@ aux-build:issue-20727.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/20727
#![crate_name="issue_20727_2"]

extern crate issue_20727;

//@ has issue_20727_2/trait.Add.html
pub trait Add<RHS = Self> {
    //@ has - '//pre[@class="rust item-decl"]' 'trait Add<RHS = Self> {'
    //@ has - '//pre[@class="rust item-decl"]' 'type Output;'
    type Output;

    //@ has - '//pre[@class="rust item-decl"]' 'fn add(self, rhs: RHS) -> Self::Output;'
    fn add(self, rhs: RHS) -> Self::Output;
}

//@ has issue_20727_2/reexport/trait.Add.html
pub mod reexport {
    //@ has - '//pre[@class="rust item-decl"]' 'trait Add<RHS = Self> {'
    //@ has - '//pre[@class="rust item-decl"]' 'type Output;'
    //@ has - '//pre[@class="rust item-decl"]' 'fn add(self, rhs: RHS) -> Self::Output;'
    pub use issue_20727::Add;
}
