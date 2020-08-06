// allow-tidy-line-length
#![deny(broken_intra_doc_links)]

/// Link to [S::assoc_fn()]
/// Link to [Default::default()]
// @has intra_link_trait_item/struct.S.html '//*[@href="https://doc.rust-lang.org/nightly/core/default/trait.Default.html#tymethod.default"]' 'Default::default()'
// @has - '//*[@href="../intra_link_trait_item/struct.S.html#method.assoc_fn"]' 'S::assoc_fn()'
pub struct S;

impl S {
    pub fn assoc_fn() {}
}
