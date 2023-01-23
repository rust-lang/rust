// aux-build:issue-20727.rs
// ignore-cross-compile

extern crate issue_20727;

pub trait Bar {}

// @has issue_20727_3/trait.Deref2.html
pub trait Deref2 {
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'trait Deref2 {'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'type Target: Bar;'
    type Target: Bar;

    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'fn deref(&self) -> Self::Target;'
    fn deref(&self) -> Self::Target;
}

// @has issue_20727_3/reexport/trait.Deref2.html
pub mod reexport {
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'trait Deref2 {'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'type Target: Bar;'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'fn deref(&self) -> Self::Target;'
    pub use issue_20727::Deref2;
}
