//@ revisions: old next
//@[next] compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]
trait Trait<'a> {
    type Out<U>;
}

impl<'a, T> Trait<'a> for T {
    type Out<U> = T;
}

type Foo = impl Sized;

fn weird_bound<X>(x: &<X as Trait<'static>>::Out<Foo>) -> X
//[old]~^ ERROR: item does not constrain
//[next]~^^ ERROR: cannot satisfy `Foo == _`
where
    for<'a> X: Trait<'a>,
    for<'a> <X as Trait<'a>>::Out<()>: Copy,
{
    let x = *x; //[old]~ ERROR: cannot move out of `*x`
    todo!();
}

fn main() {
    let _: () = weird_bound(&());
}
