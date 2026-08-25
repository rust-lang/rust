//@ run-pass
// Test equality constraints on associated types in a where clause.


pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

#[derive(PartialEq)]
pub struct Bar;

impl Foo for isize {
    type A = usize;
    fn boo(&self) -> usize { 42 }
}

impl Foo for char {
    type A = Bar;
    fn boo(&self) -> Bar { Bar }
}

fn foo_bar<I: Foo<A=Bar>>(x: I) -> Bar {
    x.boo()
}

fn foo_uint<I: Foo<A=usize>>(x: I) -> usize {
    x.boo()
}

pub fn main() {
    let a = 42;
    foo_uint(a);

    let a = 'a';
    foo_bar(a);
}
