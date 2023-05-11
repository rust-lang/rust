// Regression test for <https://github.com/rust-lang/rust/issues/107995>.

#![crate_name = "foo"]

// @has 'foo/fn.foo.html'
// @has - '//*[@class="docblock"]//a[@href="fn.bar.html"]' 'bar`'
/// A foo, see also [ bar`]
pub fn foo() {}

// @has 'foo/fn.bar.html'
// @has - '//*[@class="docblock"]' 'line Path line'
// @has - '//*[@class="docblock"]//a[@href="struct.Path.html"]' 'Path'
#[doc = "line ["]
#[doc = "Path"]
#[doc = "] line"]
pub fn bar() {}

// @has 'foo/fn.another.html'
// @has - '//*[@class="docblock"]//a[@href="struct.Path.html"]' 'Path'
/// [ `Path`]
pub fn another() {}

// @has 'foo/fn.last.html'
// @has - '//*[@class="docblock"]//a[@href="struct.Path.html"]' 'Path'
/// [ Path`]
pub fn last() {}

pub struct Path;
