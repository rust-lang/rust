//@ run-pass

#![allow(dead_code)]


trait A<T> { fn get(self) -> T; }
trait B<T, U> { fn get(self) -> (T,U); }
trait C<'a, U> { fn get(self) -> &'a U; }

mod foo {
    pub trait D<'a, T> { fn get(self) -> &'a T; }
}

fn foo1<T>(_: &(dyn A<T> + Send)) {}
fn foo2<T>(_: Box<dyn A<T> + Send + Sync>) {}
fn foo3<T>(_: Box<dyn B<isize, usize> + 'static>) {}
fn foo4<'a, T>(_: Box<dyn C<'a, T> + 'static + Send>) {}
fn foo5<'a, T>(_: Box<dyn foo::D<'a, T> + 'static + Send>) {}

pub fn main() {}
