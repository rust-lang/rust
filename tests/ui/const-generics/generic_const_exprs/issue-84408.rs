// Regression test for #84408.
//@ check-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Melon<const X: usize> {
    fn new(arr: [i32; X]) -> Self;
    fn change<T: Melon<X>>(self) -> T;
}

struct Foo([i32; 5]);
struct Bar<const A: usize, const B: usize>([i32; A + B])
where
    [(); A + B]: ;

impl Melon<5> for Foo {
    fn new(arr: [i32; 5]) -> Self {
        Foo(arr)
    }
    fn change<T: Melon<5>>(self) -> T {
        T::new(self.0)
    }
}

impl<const A: usize, const B: usize> Melon<{ A + B }> for Bar<A, B>
where
    [(); A + B]: ,
{
    fn new(arr: [i32; A + B]) -> Self {
        Bar(arr)
    }
    fn change<T: Melon<{ A + B }>>(self) -> T {
        T::new(self.0)
    }
}

fn main() {}
