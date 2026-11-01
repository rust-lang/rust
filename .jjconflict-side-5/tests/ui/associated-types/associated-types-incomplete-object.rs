// Check that the user gets an error if they omit a binding from an
// object type.

pub trait Foo {
    type A;
    type B;
    fn boo(&self) -> <Self as Foo>::A;
}

struct Bar;

impl Foo for isize {
    type A = usize;
    type B = char;
    fn boo(&self) -> usize {
        42
    }
}

pub fn main() {
    let a = &42isize as &dyn Foo<A=usize, B=char>;

    let b = &42isize as &dyn Foo<A=usize>;
    //~^ ERROR the value of the associated type `B` in `Foo` must be specified

    let c = &42isize as &dyn Foo<B=char>;
    //~^ ERROR the value of the associated type `A` in `Foo` must be specified

    let d = &42isize as &dyn Foo;
    //~^ ERROR the value of the associated types `A` and `B` in `Foo`
}
