//@ revisions: rpass1 bfail2
//@ compile-flags: -Z query-dep-graph
//@ ignore-backends: gcc

#![allow(warnings)]
#![feature(rustc_attrs)]

// Sanity check for the dirty-clean system. We add #[rustc_clean]
// attributes in places that are not checked and make sure that this causes an
// error.

fn main() {

    #[rustc_clean(except="hir_owner", cfg="bfail2")]
    //[bfail2]~^ ERROR found unchecked `#[rustc_clean]` attribute
    {
        // empty block
    }

    #[rustc_clean(cfg="bfail2")]
    //[bfail2]~^ ERROR found unchecked `#[rustc_clean]` attribute
    {
        // empty block
    }
}

struct _Struct {
    #[rustc_clean(except="hir_owner", cfg="bfail2")]
    //[bfail2]~^ ERROR found unchecked `#[rustc_clean]` attribute
    _field1: i32,

    #[rustc_clean(cfg="bfail2")]
    //[bfail2]~^ ERROR found unchecked `#[rustc_clean]` attribute
    _field2: i32,
}
