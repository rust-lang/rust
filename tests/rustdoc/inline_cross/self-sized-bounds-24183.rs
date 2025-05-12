// https://github.com/rust-lang/rust/issues/24183
#![crate_type = "lib"]
#![crate_name = "usr"]

//@ aux-crate:issue_24183=issue-24183.rs
//@ edition: 2021

//@ has usr/trait.U.html
//@ has - '//*[@class="rust item-decl"]' "pub trait U {"
//@ has - '//*[@id="method.modified"]' \
// "fn modified(self) -> Self\
// where \
//     Self: Sized"
//@ snapshot method_no_where_self_sized - '//*[@id="method.touch"]/*[@class="code-header"]'
pub use issue_24183::U;

//@ has usr/trait.S.html
//@ has - '//*[@class="rust item-decl"]' 'pub trait S: Sized {'
pub use issue_24183::S;
