// ignore-tidy-linelength
#![deny(broken_intra_doc_links)]

/// Link to [S::assoc_fn()]
/// Link to [Default::default()]
// @has trait_item/struct.S.html '//*[@href="../trait_item/struct.S.html#method.assoc_fn"]' 'S::assoc_fn()'
// @has - '//*[@href="https://doc.rust-lang.org/nightly/core/default/trait.Default.html#tymethod.default"]' 'Default::default()'
pub struct S;

impl S {
    pub fn assoc_fn() {}
}
