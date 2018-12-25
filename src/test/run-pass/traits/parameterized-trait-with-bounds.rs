// run-pass
// pretty-expanded FIXME #23616

#![allow(dead_code)]


trait A<T> { fn get(self) -> T; }
trait B<T, U> { fn get(self) -> (T,U); }
trait C<'a, U> { fn get(self) -> &'a U; }

mod foo {
    pub trait D<'a, T> { fn get(self) -> &'a T; }
}

fn foo1<T>(_: &(A<T> + Send)) {}
fn foo2<T>(_: Box<A<T> + Send + Sync>) {}
fn foo3<T>(_: Box<B<isize, usize> + 'static>) {}
fn foo4<'a, T>(_: Box<C<'a, T> + 'static + Send>) {}
fn foo5<'a, T>(_: Box<foo::D<'a, T> + 'static + Send>) {}

pub fn main() {}
