// Test interaction between unboxed closure sugar and default type
// parameters (should be exactly as if angle brackets were used).

#![feature(rustc_attrs, unboxed_closures)]
#![rustc_no_implicit_bounds]
#![allow(dead_code)]

trait Foo<T,V=T> {
    type Output;
    fn dummy(&self, t: T, v: V);
}

trait Eq<X> { fn same_types(&self, x: &X) -> bool { true } }
impl<X> Eq<X> for X { }
fn eq<A,B>() where A : Eq<B> { }

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
