// This test ensures that the syntax highlighting is correctly set on code examples
// if the `edition` attribute is used.
// Regression test for <https://github.com/rust-lang/rust/issues/148221>.

//@ edition:2024

#![crate_name = "foo"]

//@ has 'foo/fn.foo.html'
// There should be two spans: one for `kw` and one for `number`. None for `async` because
// in the 2015 edition, `async` is not a keyword.
//@ count - '//*[@class="rust rust-example-rendered"]/code/span' 2
//@ matches - '//*[@class="rust rust-example-rendered"]/code/span[@class="kw"]' '^let $'
//@ matches - '//*[@class="rust rust-example-rendered"]/code/span[@class="number"]' '^2$'
//@ has - '//*[@class="rust rust-example-rendered"]/code' 'let async = 2;'

/// ```edition2015
/// let async = 2;
/// ```
pub async fn foo() {}

//@ has 'foo/fn.another.html'
// There should be one span: `kw` (which includes all items at once). `async` is a keyword
// here since there is no edition specified for the code block, so it inherits the crate's.
//@ count - '//*[@class="rust rust-example-rendered"]/code/span' 1
//@ matches - '//*[@class="rust rust-example-rendered"]/code/span[@class="kw"]' '^async fn $'
//@ has - '//*[@class="rust rust-example-rendered"]/code' 'async fn bar() {}'

/// ```
/// async fn bar() {}
/// ```
pub fn another() {}
