// revisions: rpass1 cfail2
// compile-flags: -Z query-dep-graph

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

    #[rustc_clean(label="typeck_tables_of", cfg="cfail2")]
    pub fn y() {
        //[cfail2]~^ ERROR `typeck_tables_of(y::y)` should be clean but is not
        x::x();
    }
}

mod z {
    #[rustc_dirty(label="typeck_tables_of", cfg="cfail2")]
    pub fn z() {
        //[cfail2]~^ ERROR `typeck_tables_of(z::z)` should be dirty but is not
    }
}
