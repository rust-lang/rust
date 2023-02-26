// Test interaction between unboxed closure sugar and region
// parameters (should be exactly as if angle brackets were used
// and regions omitted).

#![feature(unboxed_closures)]
#![allow(dead_code)]

use std::marker;

trait Foo<'a,T> {
    type Output;
    fn dummy(&'a self) -> &'a (T,Self::Output);
}

trait Eq<X: ?Sized> { fn is_of_eq_type(&self, x: &X) -> bool { true } }
impl<X: ?Sized> Eq<X> for X { }
fn eq<A: ?Sized,B: ?Sized +Eq<A>>() { }

fn same_type<A,B:Eq<A>>(a: A, b: B) { }

fn test<'a,'b>() {
    // Parens are equivalent to omitting default in angle.
    eq::< dyn Foo<(isize,),Output=()>,               dyn Foo(isize)                      >();

    // Here we specify 'static explicitly in angle-bracket version.
    // Parenthesized winds up getting inferred.
    eq::< dyn Foo<'static, (isize,),Output=()>,      dyn Foo(isize)                      >();
}

fn test2(x: &dyn Foo<(isize,),Output=()>, y: &dyn Foo(isize)) {
    //~^ ERROR trait takes 1 lifetime argument but 0 lifetime arguments were supplied
    // Here, the omitted lifetimes are expanded to distinct things.
    same_type(x, y)
}

fn main() { }
