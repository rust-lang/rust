#![allow(incomplete_features)]
#![feature(generic_associated_types)]

trait SomeTrait {
    type Wrapped<A>: SomeTrait;
         //~^ ERROR: missing generics for associated type `SomeTrait::Wrapped`

    fn f() -> ();
}

fn program<W>() -> ()
where
    W: SomeTrait<Wrapped = W>,
{
    return W::f();
}

fn main() {}
