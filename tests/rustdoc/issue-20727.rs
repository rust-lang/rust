// aux-build:issue-20727.rs
// ignore-cross-compile

extern crate issue_20727;

// @has issue_20727/trait.Deref.html
pub trait Deref {
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'trait Deref {'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'type Target: ?Sized;'
    type Target: ?Sized;

    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' \
    //        "fn deref<'a>(&'a self) -> &'a Self::Target;"
    fn deref<'a>(&'a self) -> &'a Self::Target;
}

// @has issue_20727/reexport/trait.Deref.html
pub mod reexport {
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'trait Deref {'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'type Target: ?Sized;'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' \
    //      "fn deref<'a>(&'a self) -> &'a Self::Target;"
    pub use issue_20727::Deref;
}
