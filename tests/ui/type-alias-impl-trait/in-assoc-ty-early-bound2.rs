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
            //~^ ERROR expected generic lifetime parameter, found `'_`
        };
    }
}

fn main() {}
