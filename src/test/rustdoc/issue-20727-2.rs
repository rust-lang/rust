// aux-build:issue-20727.rs
// ignore-cross-compile

extern crate issue_20727;

// @has issue_20727_2/trait.Add.html
pub trait Add<RHS = Self> {
    // @has - '//*[@class="rust trait"]' 'trait Add<RHS = Self> {'
    // @has - '//*[@class="rust trait"]' 'type Output;'
    type Output;

    // @has - '//*[@class="rust trait"]' 'fn add(self, rhs: RHS) -> Self::Output;'
    fn add(self, rhs: RHS) -> Self::Output;
}

// @has issue_20727_2/reexport/trait.Add.html
pub mod reexport {
    // @has - '//*[@class="rust trait"]' 'trait Add<RHS = Self> {'
    // @has - '//*[@class="rust trait"]' 'type Output;'
    // @has - '//*[@class="rust trait"]' 'fn add(self, rhs: RHS) -> Self::Output;'
    pub use issue_20727::Add;
}
