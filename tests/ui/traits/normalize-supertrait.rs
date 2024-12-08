// Regression test for #77653
// When monomorphizing `f` we need to prove `dyn Derived<()>: Base<()>`. This
// requires us to normalize the `Base<<() as Proj>::S>` to `Base<()>` when
// comparing the supertrait `Derived<()>` to the expected trait.

//@ build-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Proj {
    type S;
}

impl Proj for () {
    type S = ();
}

impl Proj for i32 {
    type S = i32;
}

trait Base<T> {
    fn is_base(&self);
}

trait Derived<B: Proj>: Base<B::S> + Base<()> {
    fn is_derived(&self);
}

fn f<P: Proj>(obj: &dyn Derived<P>) {
    obj.is_derived();
    Base::<P::S>::is_base(obj);
    Base::<()>::is_base(obj);
}

fn main() {
    let x: fn(_) = f::<()>;
    let x: fn(_) = f::<i32>;
}
