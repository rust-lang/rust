#![feature(intrinsics)]
#![feature(no_core, lang_items)]
#![feature(rustc_attrs)]

#![no_core]
#![crate_name = "foo"]

#[lang = "sized"]
trait Sized {}

//@ has 'foo/fn.abort.html'
//@ has - '//pre[@class="rust item-decl"]' 'pub fn abort() -> !'
#[rustc_intrinsic]
pub fn abort() -> !;
//@ has 'foo/fn.unreachable.html'
//@ has - '//pre[@class="rust item-decl"]' 'pub unsafe fn unreachable() -> !'
#[rustc_intrinsic]
pub unsafe fn unreachable() -> !;
