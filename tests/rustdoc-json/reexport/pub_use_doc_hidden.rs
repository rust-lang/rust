// Regression test for <https://github.com/rust-lang/rust/issues/106379>

mod repeat_n {
    #[doc(hidden)]
    pub struct RepeatN {}
}

pub use repeat_n::RepeatN;

// @count "$.index[*][?(@.name=='pub_use_doc_hidden')].inner.items[*]" 0
// @!has "$.index[*][?(@.kind=='struct')]"
// @!has "$.index[*][?(@.kind=='import')]"
