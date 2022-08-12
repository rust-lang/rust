#![feature(no_core, auto_traits, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

pub auto trait Bar {}

// @has auto.json "$.index[*][?(@.kind=='impl')].span" null
pub struct Foo;
