#![crate_name = "foo"]
#![no_std]

// @has 'foo/fn.foo.html'
// @has - '//*[@class="docblock"]' 'inc2 x'
#[doc = include_str!("short-line.md")]
pub fn foo() {}
