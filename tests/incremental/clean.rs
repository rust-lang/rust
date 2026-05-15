//@ revisions: rpass1 bfail2
//@ compile-flags: -Z query-dep-graph
//@ ignore-backends: gcc

#![allow(warnings)]
#![feature(rustc_attrs)]

// Sanity check for the dirty-clean system. Give the opposite
// annotations that we expect to see, so that we check that errors are
// reported.

fn main() { }

mod x {
    #[cfg(rpass1)]
    pub fn x() -> usize {
        22
    }

    #[cfg(bfail2)]
    pub fn x() -> u32 {
        22
    }
}

mod y {
    use x;

    #[rustc_clean(
        except="owner,generics_of,predicates_of,type_of,fn_sig",
        cfg="bfail2",
    )]
    pub fn y() {
        //[bfail2]~^ ERROR `owner(y)` should be dirty but is not
        //[bfail2]~| ERROR `generics_of(y)` should be dirty but is not
        //[bfail2]~| ERROR `predicates_of(y)` should be dirty but is not
        //[bfail2]~| ERROR `type_of(y)` should be dirty but is not
        //[bfail2]~| ERROR `fn_sig(y)` should be dirty but is not
        //[bfail2]~| ERROR `typeck_root(y)` should be clean but is not
        x::x();
    }
}

mod z {
    #[rustc_clean(except="typeck_root", cfg="bfail2")]
    pub fn z() {
        //[bfail2]~^ ERROR `typeck_root(z)` should be dirty but is not
    }
}
