#![feature(type_alias_impl_trait)]
trait Trait<'a> {
    type Out<U>;
}

impl<'a, T> Trait<'a> for T {
    type Out<U> = T;
}

type Foo = impl Sized;
//~^ ERROR: unconstrained opaque type

fn weird_bound<X>(x: &<X as Trait<'static>>::Out<Foo>) -> X
//~^ ERROR: item does not constrain
where
    for<'a> X: Trait<'a>,
    for<'a> <X as Trait<'a>>::Out<()>: Copy,
{
    let x = *x; //~ ERROR: cannot move out of `*x`
    todo!();
}

fn main() {
    let _: () = weird_bound(&());
}
