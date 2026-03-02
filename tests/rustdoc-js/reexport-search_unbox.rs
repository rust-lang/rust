//@ aux-crate:priv:reexport_search_unbox=reexport-search_unbox.rs
//@ compile-flags: -Zunstable-options --extern equivalent

#![crate_name = "foo"]

extern crate reexport_search_unbox;

pub use reexport_search_unbox::{Inside, Out, Out1, Out2};

pub fn alpha<const N: usize, T>(_: Inside<T>) -> Out<Out1<T, N>, Out2<T, N>> {
    loop {}
}
