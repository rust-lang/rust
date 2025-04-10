// ignore-tidy-linelength
//@ edition: 2021
//@ aux-crate:suggest_trait_reexported_as_not_doc_visible_a=suggest-trait-reexported-as-not-doc-visible-a.rs
//@ aux-crate:suggest_trait_reexported_as_not_doc_visible_b=suggest-trait-reexported-as-not-doc-visible-b.rs

use suggest_trait_reexported_as_not_doc_visible_b::Bar;

fn main() {
    Bar::foo();
    //~^ ERROR: no function or associated item named `foo` found for struct `Bar` in the current scope [E0599]
}
