// Regression test for issue #67007
// Ensures that we show information about the specific regions involved

#![feature(nll)]

// Covariant over 'a, invariant over 'tcx
struct FnCtxt<'a, 'tcx: 'a>(&'a (), *mut &'tcx ());

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn use_it(&self, _: &'tcx ()) {}
}

struct Consumer<'tcx>(&'tcx ());

impl<'tcx> Consumer<'tcx> {
    fn bad_method<'a>(&self, fcx: &FnCtxt<'a, 'tcx>) {
        let other = self.use_fcx(fcx); //~ ERROR lifetime may not live long enough
        fcx.use_it(other);
    }

    fn use_fcx<'a>(&self, _: &FnCtxt<'a, 'tcx>) -> &'a () {
        &()
    }
}

fn main() {}
