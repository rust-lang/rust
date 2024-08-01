//! A regression test for #120343. The overlap error was previously
//! silenced in coherence because projecting `<() as ToUnit>::Unit`
//! failed. Then silenced the missing items error in the `ToUnit`
//! impl, causing us to not emit any errors and ICEing due to a
//! `span_delay_bug`.

trait ToUnit {
    type Unit;
}

impl<T> ToUnit for *const T {}
//~^ ERROR: not all trait items implemented

trait Overlap<T> {}

impl<T> Overlap<T> for T {}

impl<T> Overlap<<*const T as ToUnit>::Unit> for T {}

fn main() {}
