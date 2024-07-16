#![crate_name = "foo"]
#![feature(precise_capturing)]

//@ has foo/fn.two.html '//section[@id="main-content"]//pre' "-> impl Sized + use<'b, 'a>"
pub fn two<'a, 'b, 'c>() -> impl Sized + use<'b, 'a /* no 'c */> {}

//@ has foo/fn.params.html '//section[@id="main-content"]//pre' "-> impl Sized + use<'a, T, N>"
pub fn params<'a, T, const N: usize>() -> impl Sized + use<'a, T, N> {}

//@ has foo/fn.none.html '//section[@id="main-content"]//pre' "-> impl Sized + use<>"
pub fn none() -> impl Sized + use<> {}

//@ has foo/fn.first.html '//section[@id="main-content"]//pre' "-> impl use<> + Sized"
pub fn first() -> impl use<> + Sized {}
