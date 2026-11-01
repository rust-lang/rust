// This test ensures that the syntax highlighting is correctly set on the expanded
// macro to match the original macro's crate's edition.
// Regression test for <https://github.com/rust-lang/rust/issues/148221>.

//@ edition:2024
//@ aux-build:editions.rs
//@ compile-flags: -Zunstable-options --generate-macro-expansion

#![crate_name = "foo"]

#[macro_use]
extern crate editions;

//@ has 'src/foo/editions.rs.html'

// There should be one span: `kw` (which includes all items at once). `async` is a keyword
// here since it's the 2024 edition.
//@ matches - '//pre[@class="rust"]/code/span[@class="kw"]' '^async fn $'
async fn foo() {
    // There should be two spans: one for `kw` and one for `number`. None for `async` because
    // in the 2015 edition, `async` is not a keyword.
    //@ count - '//code/*[@class="expansion"]/*[@class="expanded"]/span' 2
    //@ matches - '//code/*[@class="expansion"]/*[@class="expanded"]/span[@class="kw"]' '^let $'
    //@ matches - '//code/*[@class="expansion"]/*[@class="expanded"]/span[@class="number"]' '^2$'
    //@ has - '//code/*[@class="expansion"]/*[@class="expanded"]' 'let async = 2;'
    tadam!();
}
