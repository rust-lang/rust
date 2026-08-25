#![crate_name = "user"]

//@ aux-crate:impl_sized=impl-sized.rs
//@ edition:2021

//@ has user/fn.sized.html
//@ has - '//pre[@class="rust item-decl"]' "sized(x: impl Sized) -> impl Sized"
pub use impl_sized::sized;

//@ has user/fn.sized_outlives.html
//@ has - '//pre[@class="rust item-decl"]' \
//     "sized_outlives<'a>(x: impl Sized + 'a) -> impl Sized + 'a"
pub use impl_sized::sized_outlives;

//@ has user/fn.maybe_sized.html
//@ has - '//pre[@class="rust item-decl"]' "maybe_sized(x: &impl ?Sized) -> &impl ?Sized"
pub use impl_sized::maybe_sized;

//@ has user/fn.debug_maybe_sized.html
//@ has - '//pre[@class="rust item-decl"]' \
//     "debug_maybe_sized(x: &(impl Debug + ?Sized)) -> &(impl Debug + ?Sized)"
pub use impl_sized::debug_maybe_sized;

//@ has user/fn.maybe_sized_outlives.html
//@ has - '//pre[@class="rust item-decl"]' \
//     "maybe_sized_outlives<'t>(x: &(impl ?Sized + 't)) -> &(impl ?Sized + 't)"
pub use impl_sized::maybe_sized_outlives;
