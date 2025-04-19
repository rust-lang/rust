// ignore-tidy-linelength
//@ edition: 2021
//@ aux-crate:suggest_trait_reexported_as_not_doc_visible_a=suggest-trait-reexported-as-not-doc-visible-a.rs

pub struct Bar;

impl __DocHidden::Foo for Bar {
    fn foo() {}
}

#[doc(hidden)]
pub mod __DocHidden {
    pub use suggest_trait_reexported_as_not_doc_visible_a::Foo;
}
