// This test checks that a change in a CGU does not invalidate an unrelated CGU
// during incremental ThinLTO.

//@ revisions: bfail1 bfail2 bfail3
//@ compile-flags: -Z query-dep-graph -O
//@ build-pass
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type="rlib"]

#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-foo",
                            cfg="bfail2",
                            kind="no")]
#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-foo",
                            cfg="bfail3",
                            kind="pre-lto")] // Should be "post-lto", see issue #119076

#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-bar",
                            cfg="bfail2",
                            kind="pre-lto")]
#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-bar",
                            cfg="bfail3",
                            kind="pre-lto")] // Should be "post-lto", see issue #119076

#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-baz",
                            cfg="bfail2",
                            kind="pre-lto")] // Should be "post-lto", see issue #119076
#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-baz",
                            cfg="bfail3",
                            kind="pre-lto")] // Should be "post-lto", see issue #119076
mod foo {

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

pub mod baz {
    pub fn unrelated_to_other_fns() -> u64 {
        0xbeef
    }
}
