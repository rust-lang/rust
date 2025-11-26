//! rustdoc regression test for #149288: const generic parameter types may depend on
//! other generics when `generic_const_parameter_types` is enabled.
#![allow(incomplete_features)]
#![feature(adt_const_params, generic_const_parameter_types)]
#![crate_name = "foo"]

pub struct Bar<const N: usize, const M: [u8; N]>;

pub fn takes<const N: usize, const M: [u8; N]>(_: Bar<N, M>) {}

pub fn instantiate() {
    takes(Bar::<2, { [1; 2] }>);
}

//@ has foo/struct.Bar.html '//pre[@class="rust item-decl"]' 'pub struct Bar<const N: usize, const M: [u8; N]>'
//@ has foo/fn.takes.html '//pre[@class="rust item-decl"]' 'pub fn takes<const N: usize, const M: [u8; N]>(_: Bar<N, M>)'
