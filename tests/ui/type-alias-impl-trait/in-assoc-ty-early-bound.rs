#![feature(impl_trait_in_assoc_type)]

trait Foo {
    type Assoc<'a, 'b>;
    fn bar<'a: 'a, 'b: 'b>(_: &'a ()) -> Self::Assoc<'a, 'b>;
}

impl Foo for () {
    type Assoc<'a, 'b> = impl Sized;
    fn bar<'a: 'a, 'b: 'b>(x: &'a ()) -> Self::Assoc<'a, 'b> {
        let closure = |x: &'a ()| -> Self::Assoc<'b, 'a> { x };
        //~^ ERROR expected generic lifetime parameter, found `'_`
        x
    }
}

fn main() {}
