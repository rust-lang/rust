#![feature(impl_trait_in_assoc_type)]

trait Foo {
    type Assoc<'a>;
    fn bar<'a: 'a>();
}

impl Foo for () {
    type Assoc<'a> = impl Sized;
    fn bar<'a: 'a>()
    where
        Self::Assoc<'a>:,
    {
        let _ = |x: &'a ()| {
            let _: Self::Assoc<'a> = x;
            //~^ ERROR `<() as Foo>::Assoc<'a>` captures lifetime that does not appear in bound
        };
    }
}

fn main() {}
