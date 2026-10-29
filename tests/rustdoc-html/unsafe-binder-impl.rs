//@ compile-flags: -Znext-solver

// If impls on unsafe binders are allowed, they should be listed wherever the
// inner type is.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

//@ has 'unsafe_binder_impl/struct.Local.html' '//*[@class="impl"]' 'impl Tr for &Local'
// FIXME(unsafe_binders): this impl should be listed here too.
//@ !has 'unsafe_binder_impl/struct.Local.html' '//*[@class="impl"]' "unsafe<'a>"
//@ has 'unsafe_binder_impl/trait.Tr.html' '//*[@class="impl"]' "impl Tr for unsafe<'a> &'a Local"
pub struct Local;

pub trait Tr {}

impl Tr for &Local {}
impl Tr for unsafe<'a> &'a Local {}
