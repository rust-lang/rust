//@ check-pass

#![deny(impl_trait_overcaptures)]

struct Ctxt<'tcx>(&'tcx ());

// In `compute`, we don't care that we're "overcapturing" `'tcx`
// in edition 2024, because it can be shortened at the call site
// and we know it outlives `'_`.

impl<'tcx> Ctxt<'tcx> {
    fn compute(&self) -> impl Sized + '_ {}
}

fn main() {}
