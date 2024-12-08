// Test a trait (`Bar`) with a higher-ranked supertrait.
#![allow(unconditional_recursion)]

trait Foo<'tcx> {
    fn foo(&'tcx self) -> &'tcx isize;
}

trait Bar<'ccx>: for<'tcx> Foo<'tcx> {
    fn bar(&'ccx self) -> &'ccx isize;
}

fn want_foo_for_some_tcx<'x, F: Foo<'x>>(f: &'x F) {
    want_foo_for_some_tcx(f);
    want_foo_for_any_tcx(f);
    //~^ ERROR the trait bound `for<'tcx> F: Foo<'tcx>` is not satisfied
}

fn want_foo_for_any_tcx<F: for<'tcx> Foo<'tcx>>(f: &F) {
    want_foo_for_some_tcx(f);
    want_foo_for_any_tcx(f);
}

fn want_bar_for_some_ccx<'x, B: Bar<'x>>(b: &B) {
    want_foo_for_some_tcx(b);
    want_foo_for_any_tcx(b);

    want_bar_for_some_ccx(b);
    want_bar_for_any_ccx(b);
    //~^ ERROR the trait bound `for<'ccx> B: Bar<'ccx>` is not satisfied
}

fn want_bar_for_any_ccx<B: for<'ccx> Bar<'ccx>>(b: &B) {
    want_foo_for_some_tcx(b);
    want_foo_for_any_tcx(b);

    want_bar_for_some_ccx(b);
    want_bar_for_any_ccx(b);
}

fn main() {}
