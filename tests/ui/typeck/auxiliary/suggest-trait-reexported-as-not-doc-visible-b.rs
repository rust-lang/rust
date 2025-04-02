// ignore-tidy-linelength
//@ edition: 2021
//@ aux-crate:suggest_trait_reexported_as_not_doc_visible_a=suggest-trait-reexported-as-not-doc-visible-a.rs

#[doc(hidden)]
pub use suggest_trait_reexported_as_not_doc_visible_a::TraitA;

#[doc(hidden)]
pub mod __DocHidden {
    pub use suggest_trait_reexported_as_not_doc_visible_a::TraitB;

    pub mod Inner {
        pub use suggest_trait_reexported_as_not_doc_visible_a::TraitC;
    }
}

impl suggest_trait_reexported_as_not_doc_visible_a::TraitA for Foo {
    fn a() {}
}

impl __DocHidden::TraitB for Foo {
    fn b() {}
}

impl __DocHidden::Inner::TraitC for Foo {
    fn c() {}
}

pub struct Foo;
