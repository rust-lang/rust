//! Check that associated items can be marked as lang items, so that they don't have to be looked up
//! by name or by definition order indirectly.
//!
//! This test is not *quite* high-fidelity: it checks that you can use lang items on associated
//! items by looking at the error message *as a proxy*. That is, the error message is about
//! undefined lang items and not invalid attribute target, indicating that it has reached lang item
//! machinery (which is relying on knowing the implementation detail). However, it's annoying to
//! write a full-fidelity test for this, so I think this is acceptable even though it's not *great*.
//!
//! This was implemented in <https://github.com/rust-lang/rust/pull/72559> to help with
//! <https://github.com/rust-lang/rust/issues/70718>, which is itself relevant for e.g. `Fn::Output`
//! or `Future::Output` or specific use cases like [Use `T`'s discriminant type in
//! `mem::Discriminant<T>` instead of `u64`](https://github.com/rust-lang/rust/pull/70705).

#![feature(lang_items)]

trait Foo {
    #[lang = "dummy_lang_item_1"] //~ ERROR definition
    fn foo() {}

    #[lang = "dummy_lang_item_2"] //~ ERROR definition
    fn bar();

    #[lang = "dummy_lang_item_3"] //~ ERROR definition
    type MyType;
}

struct Bar;

impl Bar {
    #[lang = "dummy_lang_item_4"] //~ ERROR definition
    fn test() {}
}

fn main() {}
