// aux-build:issue-20727.rs
// ignore-cross-compile

extern crate issue_20727;

// @has issue_20727_4/trait.Index.html
pub trait Index<Idx: ?Sized> {
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'trait Index<Idx: ?Sized> {'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'type Output: ?Sized'
    type Output: ?Sized;

    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' \
    //        'fn index(&self, index: Idx) -> &Self::Output'
    fn index(&self, index: Idx) -> &Self::Output;
}

// @has issue_20727_4/trait.IndexMut.html
pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' \
    //        'trait IndexMut<Idx: ?Sized>: Index<Idx> {'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' \
    //        'fn index_mut(&mut self, index: Idx) -> &mut Self::Output;'
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}

pub mod reexport {
    // @has issue_20727_4/reexport/trait.Index.html
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'trait Index<Idx>where Idx: ?Sized,{'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'type Output: ?Sized'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' \
    //        'fn index(&self, index: Idx) -> &Self::Output'
    pub use issue_20727::Index;

    // @has issue_20727_4/reexport/trait.IndexMut.html
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' \
    //        'trait IndexMut<Idx>: Index<Idx>where Idx: ?Sized,{'
    // @has - '//div[@class="item-decl"]/pre[@class="rust"]' \
    //        'fn index_mut(&mut self, index: Idx) -> &mut Self::Output;'
    pub use issue_20727::IndexMut;
}
