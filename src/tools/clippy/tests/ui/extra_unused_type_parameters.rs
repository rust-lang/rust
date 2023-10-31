//@run-rustfix
//@aux-build:proc_macros.rs:proc-macro

#![allow(unused, clippy::needless_lifetimes)]
#![warn(clippy::extra_unused_type_parameters)]

extern crate proc_macros;
use proc_macros::with_span;

fn unused_ty<T>(x: u8) {
    unimplemented!()
}

fn unused_multi<T, U>(x: u8) {
    unimplemented!()
}

fn unused_with_lt<'a, T>(x: &'a u8) {
    unimplemented!()
}

fn used_ty<T>(x: T, y: u8) {}

fn used_ref<'a, T>(x: &'a T) {}

fn used_ret<T: Default>(x: u8) -> T {
    T::default()
}

fn unused_bounded<T: Default, U, V: Default>(x: U) {
    unimplemented!();
}

fn some_unused<A, B, C, D: Iterator<Item = (B, C)>, E>(b: B, c: C) {
    unimplemented!();
}

fn used_opaque<A>(iter: impl Iterator<Item = A>) -> usize {
    iter.count()
}

fn used_ret_opaque<A>() -> impl Iterator<Item = A> {
    std::iter::empty()
}

fn used_vec_box<T>(x: Vec<Box<T>>) {}

fn used_body<T: Default + ToString>() -> String {
    T::default().to_string()
}

fn used_closure<T: Default + ToString>() -> impl Fn() {
    || println!("{}", T::default().to_string())
}

struct S;

impl S {
    fn unused_ty_impl<T>(&self) {
        unimplemented!()
    }
}

// Don't lint on trait methods
trait Foo {
    fn bar<T>(&self);
}

impl Foo for S {
    fn bar<T>(&self) {}
}

fn skip_index<A, Iter>(iter: Iter, index: usize) -> impl Iterator<Item = A>
where
    Iter: Iterator<Item = A>,
{
    iter.enumerate()
        .filter_map(move |(i, a)| if i == index { None } else { Some(a) })
}

fn unused_opaque<A, B>(dummy: impl Default) {
    unimplemented!()
}

mod unexported_trait_bounds {
    mod private {
        pub trait Private {}
    }

    fn priv_trait_bound<T: private::Private>() {
        unimplemented!();
    }

    fn unused_with_priv_trait_bound<T: private::Private, U>() {
        unimplemented!();
    }
}

mod issue10319 {
    fn assert_send<T: Send>() {}

    fn assert_send_where<T>()
    where
        T: Send,
    {
    }
}

with_span!(
    span

    fn should_not_lint<T>(x: u8) {
        unimplemented!()
    }
);

fn main() {}
