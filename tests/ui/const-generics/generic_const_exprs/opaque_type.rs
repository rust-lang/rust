#![feature(generic_const_exprs, type_alias_impl_trait)]
#![allow(incomplete_features)]

type Foo = impl Sized;

fn with_bound<const N: usize>() -> Foo
where
    [u8; (N / 2) as usize]: Sized,
{
    let _: [u8; (N / 2) as Foo] = [0; (N / 2) as usize];
    //~^ ERROR mismatched types
    //~| ERROR non-primitive cast: `usize` as `Foo`
    todo!()
}

fn main() {
    with_bound::<4>();
}
