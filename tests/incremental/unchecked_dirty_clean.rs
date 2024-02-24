//@ revisions: rpass1 cfail2
//@ compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]

// Sanity check for the dirty-clean system. We add #[rustc_clean]
// attributes in places that are not checked and make sure that this causes an
// error.

fn main() {

    #[rustc_clean(except="hir_owner", cfg="cfail2")]
    //[cfail2]~^ ERROR found unchecked `#[rustc_clean]` attribute
    {
        // empty block
    }

    #[rustc_clean(cfg="cfail2")]
    //[cfail2]~^ ERROR found unchecked `#[rustc_clean]` attribute
    {
        // empty block
    }
}

struct _Struct {
    #[rustc_clean(except="hir_owner", cfg="cfail2")]
    //[cfail2]~^ ERROR found unchecked `#[rustc_clean]` attribute
    _field1: i32,

    #[rustc_clean(cfg="cfail2")]
    //[cfail2]~^ ERROR found unchecked `#[rustc_clean]` attribute
    _field2: i32,
}
