#![feature(intrinsics)]
#![feature(no_core)]
#![feature(rustc_attrs)]

#![no_core]
#![crate_name = "foo"]

//@ has 'foo/fn.abort.html'
//@ has - '//pre[@class="rust item-decl"]' 'pub fn abort() -> !'
#[rustc_intrinsic]
#[rustc_intrinsic_must_be_overridden]
pub fn abort() -> ! {
    loop {}
}
//@ has 'foo/fn.unreachable.html'
//@ has - '//pre[@class="rust item-decl"]' 'pub unsafe fn unreachable() -> !'
#[rustc_intrinsic]
#[rustc_intrinsic_must_be_overridden]
pub unsafe fn unreachable() -> ! {
    loop {}
}
