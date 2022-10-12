// check-pass

#![feature(generators, generator_trait)]
#![feature(type_alias_impl_trait)]

trait Trait {}

impl<T> Trait for T {}

type Foo<'c> = impl Trait + 'c;
fn foo<'a>(rng: &'a ()) -> Foo<'a> {
    fn helper<'b>(rng: &'b ()) -> impl 'b + Trait {
        rng
    }

    helper(rng)
}

fn main() {
}
