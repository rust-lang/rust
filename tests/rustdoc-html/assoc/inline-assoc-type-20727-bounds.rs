//@ aux-build:issue-20727.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/20727
#![crate_name="issue_20727"]

extern crate issue_20727;

//@ has issue_20727/trait.Deref.html
pub trait Deref {
    //@ has - '//pre[@class="rust item-decl"]' 'trait Deref {'
    //@ has - '//pre[@class="rust item-decl"]' 'type Target: ?Sized;'
    type Target: ?Sized;

    //@ has - '//pre[@class="rust item-decl"]' \
    //        "fn deref<'a>(&'a self) -> &'a Self::Target;"
    fn deref<'a>(&'a self) -> &'a Self::Target;
}

//@ has issue_20727/reexport/trait.Deref.html
pub mod reexport {
    //@ has - '//pre[@class="rust item-decl"]' 'trait Deref {'
    //@ has - '//pre[@class="rust item-decl"]' 'type Target: ?Sized;'
    //@ has - '//pre[@class="rust item-decl"]' \
    //      "fn deref<'a>(&'a self) -> &'a Self::Target;"
    pub use issue_20727::Deref;
}
