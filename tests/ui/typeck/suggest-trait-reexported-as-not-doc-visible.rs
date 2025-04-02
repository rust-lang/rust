// ignore-tidy-linelength
//@ edition: 2021
//@ aux-crate:suggest_trait_reexported_as_not_doc_visible_a=suggest-trait-reexported-as-not-doc-visible-a.rs
//@ aux-crate:suggest_trait_reexported_as_not_doc_visible_b=suggest-trait-reexported-as-not-doc-visible-b.rs

use suggest_trait_reexported_as_not_doc_visible_b::Foo;

fn main() {
    Foo::a();
    //~ no function or associated item named `a` found for struct `Foo` in the current scope [E0599]
    Foo::b();
    Foo::c();
}
