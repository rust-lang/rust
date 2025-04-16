//@ known-bug: #137468
//@ compile-flags: -Copt-level=0 -Zmir-enable-passes=+GVN -Zvalidate-mir
trait Supertrait<T> {}

trait Identity {
    type Selff;
}

trait Trait<P>: Supertrait<()> + Supertrait<<P as Identity>::Selff> {}

impl<P> Trait<P> for () {}

fn main() {
    let x: &dyn Trait<()> = &();
    let x: &dyn Supertrait<()> = x;
}
