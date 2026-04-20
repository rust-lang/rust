// This test checks that the LTO phase is re-done for CGUs that import something
// via ThinLTO and that imported thing changes while the definition of the CGU
// stays untouched.

//@ revisions: bfail1 bfail2 bfail3
//@ compile-flags: -Z query-dep-graph -O
//@ build-pass
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type="rlib"]

#![rustc_expected_cgu_reuse(module="cgu_invalidated_via_import-foo",
                            cfg="bfail2",
                            kind="no")]
#![rustc_expected_cgu_reuse(module="cgu_invalidated_via_import-foo",
                            cfg="bfail3",
                            kind="pre-lto")] // Should be "post-lto", see issue #119076

#![rustc_expected_cgu_reuse(module="cgu_invalidated_via_import-bar",
                            cfg="bfail2",
                            kind="pre-lto")]
#![rustc_expected_cgu_reuse(module="cgu_invalidated_via_import-bar",
                            cfg="bfail3",
                            kind="pre-lto")] // Should be "post-lto", see issue #119076

mod foo {

    // Trivial functions like this one are imported very reliably by ThinLTO.
    #[cfg(bfail1)]
    pub fn inlined_fn() -> u32 {
        1234
    }

    #[cfg(not(bfail1))]
    pub fn inlined_fn() -> u32 {
        // See `cgu_keeps_identical_fn.rs` for why this is different
        // from the other version of this function.
        12345
    }
}

pub mod bar {
    use foo::inlined_fn;

    pub fn caller() -> u32 {
        inlined_fn()
    }
}
