// aux-build:issue-20727.rs
// ignore-cross-compile

extern crate issue_20727;

// @has issue_20727/trait.Deref.html
pub trait Deref {
    // @has - '//*[@class="rust trait"]' 'trait Deref {'
    // @has - '//*[@class="rust trait"]' 'type Target: ?Sized;'
    type Target: ?Sized;

    // @has - '//*[@class="rust trait"]' \
    //        "fn deref<'a>(&'a self) -> &'a Self::Target;"
    fn deref<'a>(&'a self) -> &'a Self::Target;
}

// @has issue_20727/reexport/trait.Deref.html
pub mod reexport {
    // @has - '//*[@class="rust trait"]' 'trait Deref {'
    // @has - '//*[@class="rust trait"]' 'type Target: ?Sized;'
    // @has - '//*[@class="rust trait"]' \
    //      "fn deref(&'a self) -> &'a Self::Target;"
    pub use issue_20727::Deref;
}
