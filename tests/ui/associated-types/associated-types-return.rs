// run-pass
// Test equality constraints on associated types in a where clause.


pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

#[derive(PartialEq, Debug)]
pub struct Bar;

impl Foo for isize {
    type A = usize;
    fn boo(&self) -> usize { 42 }
}

impl Foo for Bar {
    type A = isize;
    fn boo(&self) -> isize { 43 }
}

impl Foo for char {
    type A = Bar;
    fn boo(&self) -> Bar { Bar }
}

fn foo1<I: Foo<A=Bar>>(x: I) -> Bar {
    x.boo()
}

fn foo2<I: Foo>(x: I) -> <I as Foo>::A {
    x.boo()
}

pub fn main() {
    let a = 42;
    assert_eq!(foo2(a), 42);

    let a = Bar;
    assert_eq!(foo2(a), 43);

    let a = 'a';
    foo1(a);
    assert_eq!(foo2(a), Bar);
}
