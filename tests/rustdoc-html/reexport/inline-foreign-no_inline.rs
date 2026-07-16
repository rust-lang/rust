// This test ensures that an inlined foreign item with `#[doc(no_inline)]` is
// not inlined.
// This is a regression test for <https://github.com/rust-lang/rust/issues/92379>.

//@ aux-build: inline-foreign-no_inline.rs

#![crate_name = "foo"]

extern crate inline_foreign_no_inline;

// Since we cannot inline `inline_foreign_no_inline` because it has `no_inline`, there
// should have no other items than "Module".
//@ has 'foo/index.html'
//@ count - '//*[@id="main-content"]/h2' 1
//@ has - '//*[@id="main-content"]/h2' 'Module'
//@ has - '//*[@id="main-content"]/dl[@class="item-table"]/dt' 'dep'

//@ has 'foo/dep/index.html'
//@ has - '//*[@class="item-table reexports"]/dt' 'pub use dep::Future;'
//@ has - '//*[@class="item-table reexports"]/dt' 'pub use dep::FutureExt as _;'
#[doc(inline)]
pub use inline_foreign_no_inline as dep;
