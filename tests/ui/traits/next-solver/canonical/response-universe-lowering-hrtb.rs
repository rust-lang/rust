//@ check-pass
//@ compile-flags: -Znext-solver=globally

use std::marker::PhantomData;

trait Lift<I> {
    type Lifted;
}

trait Print<P> {}

#[derive(Copy, Clone)]
struct Tcx<'tcx>(PhantomData<&'tcx ()>);

#[derive(Copy, Clone)]
struct Region<I>(I);

struct Printer<'a, 'tcx>(PhantomData<(&'a (), &'tcx ())>);

// Mirrors the higher-ranked blanket bound used by rustc's `IrPrint` implementation.
trait IrPrint<T> {
    fn print(value: &T);
}

impl<T> IrPrint<T> for Tcx<'_>
where
    T: Copy + for<'a, 'tcx> Lift<Tcx<'tcx>, Lifted: Print<Printer<'a, 'tcx>>>,
{
    fn print(_: &T) {}
}

impl<'from, 'tcx> Lift<Tcx<'tcx>> for Region<Tcx<'from>> {
    type Lifted = Region<Tcx<'tcx>>;
}

impl<'a, 'tcx> Print<Printer<'a, 'tcx>> for Region<Tcx<'tcx>> {}

trait Interner: Copy + IrPrint<Region<Self>> {}

impl Interner for Tcx<'_> {}

impl<I: Interner> std::fmt::Display for Region<I> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <I as IrPrint<Region<I>>>::print(self);
        Ok(())
    }
}

fn main() {
    println!("{}", Region(Tcx(PhantomData)));
}
