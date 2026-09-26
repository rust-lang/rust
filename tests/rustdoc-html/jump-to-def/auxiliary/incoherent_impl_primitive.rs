#![feature(rustc_attrs)]
#![allow(internal_features)]

impl<T> [T] {
    #[rustc_allow_incoherent_impl]
    pub fn f(&self) {}
}
