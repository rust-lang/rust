// Because inner and outer attributes don't have the same scope for a module, we need
// to also apply the logic to the other module attributes which can contain intra-doc
// links.
// Regression test for <https://github.com/rust-lang/rust/issues/164459>.

#![deny(rustdoc::broken_intra_doc_links)]

pub mod a {
    //!
    #![deprecated = "[](TypeAlias::x)"]
    //~^^ ERROR broken_intra_doc_links
}
