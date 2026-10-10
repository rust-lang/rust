//@ revisions: old next
//@[next] compile-flags: -Znext-solver

// In the next solver, the opaque was previously defined by using the where-bound when checking
// whether the alias is `Sized`, constraining the opaque. Instead, the alias-bound is now used,
// which means the opaque is never constrained.

#![feature(type_alias_impl_trait)]
trait Trait<'a> {
    type Out<U>;
}

impl<'a, T> Trait<'a> for T {
    type Out<U> = T;
}

type Foo = impl Sized;

#[define_opaque(Foo)]
fn weird_bound<X>(x: &<X as Trait<'static>>::Out<Foo>) -> X
where
    for<'a> X: Trait<'a>,
    for<'a> <X as Trait<'a>>::Out<()>: Copy,
{
    let x = *x; //[old]~ ERROR: cannot move out of `*x`
    //[next]~^ ERROR: type annotations needed
    todo!();
}

fn main() {
    let _: () = weird_bound(&());
}
