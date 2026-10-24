//@ aux-build:issue-20727.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/20727
#![crate_name="issue_20727_4"]

extern crate issue_20727;

//@ has issue_20727_4/trait.Index.html
pub trait Index<Idx: ?Sized> {
    //@ has - '//pre[@class="rust item-decl"]' 'trait Index<Idx: ?Sized> {'
    //@ has - '//pre[@class="rust item-decl"]' 'type Output: ?Sized'
    type Output: ?Sized;

    //@ has - '//pre[@class="rust item-decl"]' \
    //        'fn index(&self, index: Idx) -> &Self::Output'
    fn index(&self, index: Idx) -> &Self::Output;
}

//@ has issue_20727_4/trait.IndexMut.html
pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    //@ has - '//pre[@class="rust item-decl"]' \
    //        'trait IndexMut<Idx: ?Sized>: Index<Idx> {'
    //@ has - '//pre[@class="rust item-decl"]' \
    //        'fn index_mut(&mut self, index: Idx) -> &mut Self::Output;'
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}

pub mod reexport {
    //@ has issue_20727_4/reexport/trait.Index.html
    //@ has - '//pre[@class="rust item-decl"]' 'trait Index<Idx>where Idx: ?Sized,{'
    //@ has - '//pre[@class="rust item-decl"]' 'type Output: ?Sized'
    //@ has - '//pre[@class="rust item-decl"]' \
    //        'fn index(&self, index: Idx) -> &Self::Output'
    pub use issue_20727::Index;

    //@ has issue_20727_4/reexport/trait.IndexMut.html
    //@ has - '//pre[@class="rust item-decl"]' \
    //        'trait IndexMut<Idx>: Index<Idx>where Idx: ?Sized,{'
    //@ has - '//pre[@class="rust item-decl"]' \
    //        'fn index_mut(&mut self, index: Idx) -> &mut Self::Output;'
    pub use issue_20727::IndexMut;
}
