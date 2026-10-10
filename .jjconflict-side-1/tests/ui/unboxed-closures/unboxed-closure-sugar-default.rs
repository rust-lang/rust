// Test interaction between unboxed closure sugar and default type
// parameters (should be exactly as if angle brackets were used).

#![feature(unboxed_closures)]
#![allow(dead_code)]

trait Foo<T,V=T> {
    type Output;
    fn dummy(&self, t: T, v: V);
}

trait Eq<X: ?Sized> { fn same_types(&self, x: &X) -> bool { true } }
impl<X: ?Sized> Eq<X> for X { }
fn eq<A: ?Sized,B: ?Sized>() where A : Eq<B> { }

fn test<'a,'b>() {
    // Parens are equivalent to omitting default in angle.
    eq::<dyn Foo<(isize,), Output=()>, dyn Foo(isize)>();

    // In angle version, we supply something other than the default
    eq::<dyn Foo<(isize,), isize, Output=()>, dyn Foo(isize)>();
    //~^ ERROR E0277

    // Supply default explicitly.
    eq::<dyn Foo<(isize,), (isize,), Output=()>, dyn Foo(isize)>();
}

fn main() { }
