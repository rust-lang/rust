// Regression test for https://github.com/rust-lang/rust/issues/165186

//@ aux-crate:gca_assoc_item_sig=gca-assoc-item-sig.rs
//@ compile-flags: -Znext-solver=globally
//@ edition:2021

#![crate_name = "user"]

//@ has 'user/struct.S.html'
//@ has - '//*[@id="method.f"]' 'pub fn f<const N: usize>(_: [u8; gca_assoc_item_sig::P::<N>])'
//@ has - '//*[@id="associatedconstant.C"]' \
// 'pub const C<const N: usize>: [u8; gca_assoc_item_sig::P::<N>]'
//@ has - '//*[@id="associatedtype.T"]' 'type T<const N: usize> = [u8; gca_assoc_item_sig::P::<N>]'
pub use gca_assoc_item_sig::S;
