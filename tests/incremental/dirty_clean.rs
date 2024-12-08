//@ revisions: rpass1 cfail2
//@ compile-flags: -Z query-dep-graph

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

    #[cfg(cfail2)]
    pub fn x() -> u32 {
        22
    }
}

mod y {
    use x;

    #[rustc_clean(
        except="opt_hir_owner_nodes,generics_of,predicates_of,type_of,fn_sig",
        cfg="cfail2",
    )]
    pub fn y() {
        //[cfail2]~^ ERROR `opt_hir_owner_nodes(y)` should be dirty but is not
        //[cfail2]~| ERROR `generics_of(y)` should be dirty but is not
        //[cfail2]~| ERROR `predicates_of(y)` should be dirty but is not
        //[cfail2]~| ERROR `type_of(y)` should be dirty but is not
        //[cfail2]~| ERROR `fn_sig(y)` should be dirty but is not
        //[cfail2]~| ERROR `typeck(y)` should be clean but is not
        x::x();
    }
}

mod z {
    #[rustc_clean(except="typeck", cfg="cfail2")]
    pub fn z() {
        //[cfail2]~^ ERROR `typeck(z)` should be dirty but is not
    }
}
