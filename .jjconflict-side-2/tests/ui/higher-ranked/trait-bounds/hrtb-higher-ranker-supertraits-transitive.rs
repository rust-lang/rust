// Test HRTB supertraits with several levels of expansion required.

trait Foo<'tcx>
{
    fn foo(&'tcx self) -> &'tcx isize;
}

trait Bar<'ccx>
    : for<'tcx> Foo<'tcx>
{
    fn bar(&'ccx self) -> &'ccx isize;
}

trait Baz
    : for<'ccx> Bar<'ccx>
{
    fn dummy(&self);
}

trait Qux
    : Bar<'static>
{
    fn dummy(&self);
}

fn want_foo_for_any_tcx<F>(f: &F)
    where F : for<'tcx> Foo<'tcx>
{
}

fn want_bar_for_any_ccx<B>(b: &B)
    where B : for<'ccx> Bar<'ccx>
{
}

fn want_baz<B>(b: &B)
    where B : Baz
{
    want_foo_for_any_tcx(b);
    want_bar_for_any_ccx(b);
}

fn want_qux<B>(b: &B)
    where B : Qux
{
    want_foo_for_any_tcx(b);
    want_bar_for_any_ccx(b); //~ ERROR
}

fn main() {}
