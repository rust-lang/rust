#![allow(unused, clippy::needless_lifetimes)]
#![warn(clippy::extra_unused_type_parameters)]

fn unused_ty<T>(x: u8) {}

fn unused_multi<T, U>(x: u8) {}

fn unused_with_lt<'a, T>(x: &'a u8) {}

fn used_ty<T>(x: T, y: u8) {}

fn used_ref<'a, T>(x: &'a T) {}

fn used_ret<T: Default>(x: u8) -> T {
    T::default()
}

fn unused_bounded<T: Default, U>(x: U) {
    unimplemented!();
}

fn unused_where_clause<T, U>(x: U)
where
    T: Default,
{
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
    fn unused_ty_impl<T>(&self) {}
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

fn unused_opaque<A, B>(dummy: impl Default) {}

mod issue10319 {
    fn assert_send<T: Send>() {}

    fn assert_send_where<T>()
    where
        T: Send,
    {
    }
}

fn main() {}
