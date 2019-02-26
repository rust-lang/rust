// aux-build:macros.rs
// build-aux-docs

#![feature(macro_test)]

#![crate_name = "foo"]

extern crate macros;

// @has foo/index.html '//*[@class="docblock-short"]/span[@class="stab deprecated"]' Deprecated
// @has - '//*[@class="docblock-short"]/span[@class="stab unstable"]' Experimental

// @has foo/macro.my_macro.html
// @has - '//*[@class="docblock"]' 'docs for my_macro'
// @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.2.3: text'
// @has - '//*[@class="stab unstable"]' 'macro_test'
// @has - '//a/@href' '../src/macros/macros.rs.html#9-11'
pub use macros::my_macro;
