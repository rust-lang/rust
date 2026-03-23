#![feature(intrinsics)]
#![feature(no_core, lang_items)]
#![feature(rustc_attrs)]

#![no_core]
#![crate_name = "foo"]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "size_of_val"]
pub trait SizeOfVal: PointeeSized {}

#[lang = "sized"]
pub trait Sized: SizeOfVal {}

//@ has 'foo/fn.abort.html'
//@ has - '//pre[@class="rust item-decl"]' 'pub fn abort() -> !'
#[rustc_intrinsic]
pub fn abort() -> !;
//@ has 'foo/fn.unreachable.html'
//@ has - '//pre[@class="rust item-decl"]' 'pub unsafe fn unreachable() -> !'
#[rustc_intrinsic]
pub unsafe fn unreachable() -> !;
