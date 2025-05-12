//@ revisions:rpass1 rpass2
//@ compile-flags: -Z query-dep-graph

// We use #[inline(always)] to ensure that the downstream crate
// will always load the MIR for these functions

#![feature(rustc_attrs)]

#[allow(unused)]
macro_rules! first_macro {
    () => {
        println!("New call!");
    }
}

#[rustc_clean(except="opt_hir_owner_nodes,typeck,optimized_mir,promoted_mir", cfg="rpass2")]
#[inline(always)]
pub fn changed_fn() {
    // This will cause additional hygiene to be generate,
    // which will cause the SyntaxContext/ExpnId raw ids to be
    // different when we write out `my_fn` to the crate metadata.
    #[cfg(rpass2)]
    first_macro!();
}

macro_rules! print_loc {
    () => {
        println!("Caller loc: {}", std::panic::Location::caller());
    }
}

#[rustc_clean(cfg="rpass2")]
#[inline(always)]
pub fn unchanged_fn() {
    print_loc!();
}
