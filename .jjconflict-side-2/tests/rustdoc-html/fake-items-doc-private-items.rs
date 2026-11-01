//! Test checking that these `_` constants don't show up
//@ compile-flags: --document-private-items

//@ !has foo/constant._.html

// both PUB_CONST and PRIV_CONST should show up, but not the anon consts
//@ count foo/index.html '//*[@class="constant"]' 2
#![crate_name = "foo"]

#![feature(rustdoc_internals)]
#![feature(rustc_attrs)]

//@ has foo/attribute.no_mangle.html '//section[@id="main-content"]//div[@class="docblock"]//p' 'hello attr'
#[doc(attribute = "no_mangle")]
/// hello attr
const _: () = ();

//@ has foo/keyword.match.html '//section[@id="main-content"]//div[@class="docblock"]//p' 'hello kw'
#[doc(keyword = "match")]
/// hello kw
const _: () = ();

//@ has foo/primitive.i128.html '//section[@id="main-content"]//div[@class="docblock"]//p' 'hello prim'
#[rustc_doc_primitive = "i128"]
/// hello prim
const _: () = ();

/// regular anon const
const _: () = ();

//@ has foo/constant.PUB_CONST.html
/// woop
pub const PUB_CONST: i32 = 42;

//@ has foo/constant.PRIV_CONST.html
/// woop2
const PRIV_CONST: i32 = 42;
