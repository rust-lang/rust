#![feature(intrinsics, rustc_attrs)]
#![feature(staged_api)]

#![crate_name = "foo"]
#![stable(since="1.0.0", feature="rust1")]

//@ has 'foo/fn.transmute.html'
//@ has - '//pre[@class="rust item-decl"]' 'pub const unsafe fn transmute<T, U>(_: T) -> U'
#[stable(since="1.0.0", feature="rust1")]
#[rustc_const_stable(feature = "const_transmute", since = "1.56.0")]
#[rustc_intrinsic]
pub const unsafe fn transmute<T, U>(_: T) -> U;

//@ has 'foo/fn.unreachable.html'
//@ has - '//pre[@class="rust item-decl"]' 'pub unsafe fn unreachable() -> !'
#[stable(since="1.0.0", feature="rust1")]
#[rustc_intrinsic]
pub unsafe fn unreachable() -> !;

extern "C" {
    //@ has 'foo/fn.needs_drop.html'
    //@ has - '//pre[@class="rust item-decl"]' 'pub unsafe extern "C" fn needs_drop() -> !'
    #[stable(since="1.0.0", feature="rust1")]
    pub fn needs_drop() -> !;
}
