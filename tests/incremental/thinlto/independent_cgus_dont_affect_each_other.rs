// This test checks that a change in a CGU does not invalidate an unrelated CGU
// during incremental ThinLTO.

// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -O
// build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]
#![crate_type="rlib"]

#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-foo",
                            cfg="cfail2",
                            kind="no")]
#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-foo",
                            cfg="cfail3",
                            kind="post-lto")]

#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-bar",
                            cfg="cfail2",
                            kind="pre-lto")]
#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-bar",
                            cfg="cfail3",
                            kind="post-lto")]

#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-baz",
                            cfg="cfail2",
                            kind="post-lto")]
#![rustc_expected_cgu_reuse(module="independent_cgus_dont_affect_each_other-baz",
                            cfg="cfail3",
                            kind="post-lto")]
mod foo {

    #[cfg(cfail1)]
    pub fn inlined_fn() -> u32 {
        1234
    }

    #[cfg(not(cfail1))]
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
