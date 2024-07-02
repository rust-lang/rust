#![feature(type_alias_impl_trait)]
trait Trait<'a> {
    type Out<U>;
}

impl<'a, T> Trait<'a> for T {
    type Out<U> = T;
}

type Foo = impl Sized;

fn weird_bound<X>(x: &<X as Trait<'static>>::Out<Foo>) -> X
where
    for<'a> X: Trait<'a>,
    for<'a> <X as Trait<'a>>::Out<()>: Copy,
{
    let x = *x; //~ ERROR: cannot move out of `*x`
    //~^ ERROR: cannot move a value of type `<X as Trait<'_>>::Out<Foo>`
    todo!();
}

fn main() {
    let _: () = weird_bound(&());
}
