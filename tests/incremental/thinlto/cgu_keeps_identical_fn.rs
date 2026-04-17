// This test is almost identical to `cgu_invalided_via_import`, except that
// the two versions of `inline_fn` are identical. Neither version of `inlined_fn`
// ends up with any spans in its LLVM bitecode, so LLVM is able to skip
// re-building any modules which import 'inlined_fn'

//@ revisions: bfail1 bfail2 bfail3
//@ compile-flags: -Z query-dep-graph -O
//@ build-pass
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type = "rlib"]
#![rustc_expected_cgu_reuse(
    module = "cgu_keeps_identical_fn-foo",
    cfg = "bfail2",
    kind = "pre-lto"
)]
#![rustc_expected_cgu_reuse(
    module = "cgu_keeps_identical_fn-foo",
    cfg = "bfail3",
    kind = "pre-lto" // Should be "post-lto", see issue #119076
)]
#![rustc_expected_cgu_reuse(
    module = "cgu_keeps_identical_fn-bar",
    cfg = "bfail2",
    kind = "pre-lto" // Should be "post-lto", see issue #119076
)]
#![rustc_expected_cgu_reuse(
    module = "cgu_keeps_identical_fn-bar",
    cfg = "bfail3",
    kind = "pre-lto" // Should be "post-lto", see issue #119076
)]

mod foo {

    // Trivial functions like this one are imported very reliably by ThinLTO.
    #[cfg(bfail1)]
    pub fn inlined_fn() -> u32 {
        1234
    }

    #[cfg(not(bfail1))]
    pub fn inlined_fn() -> u32 {
        1234
    }
}

pub mod bar {
    use foo::inlined_fn;

    pub fn caller() -> u32 {
        inlined_fn()
    }
}
