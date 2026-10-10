//@ aux-build:issue-13698.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/13698
#![crate_name="issue_13698"]

extern crate issue_13698;

pub struct Foo;
//@ has 'issue_13698/type.Foo.html'
// There is only one visible trait impl method (from the `Foo` trait).
//@ count - '//*[@id="trait-implementations-list"]//*[@class="method trait-impl"]' 1
//@ has - '//*[@id="trait-implementations-list"]//*[@class="method trait-impl"]' \
//        'fn not_hidden(&self)'
impl issue_13698::Foo for Foo {}

pub trait Bar {
    #[doc(hidden)]
    fn bar(&self) {}
}

impl Bar for Foo {}
