// This test is almost identical to `cgu_invalided_via_import`, except that
// the two versions of `inline_fn` are identical. Neither version of `inlined_fn`
// ends up with any spans in its LLVM bitecode, so LLVM is able to skip
// re-building any modules which import 'inlined_fn'

//@ revisions: bpass1 bpass2 bpass3
//@ compile-flags: -Z query-dep-graph -O
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type = "rlib"]
#![rustc_expected_cgu_reuse(
    module = "cgu_keeps_identical_fn-foo",
    cfg = "bpass2",
    kind = "pre-lto"
)]
#![rustc_expected_cgu_reuse(
    module = "cgu_keeps_identical_fn-foo",
    cfg = "bpass3",
    kind = "pre-lto" // Should be "post-lto", see issue #119076
)]
#![rustc_expected_cgu_reuse(
    module = "cgu_keeps_identical_fn-bar",
    cfg = "bpass2",
    kind = "pre-lto" // Should be "post-lto", see issue #119076
)]
#![rustc_expected_cgu_reuse(
    module = "cgu_keeps_identical_fn-bar",
    cfg = "bpass3",
    kind = "pre-lto" // Should be "post-lto", see issue #119076
)]

mod foo {

    // Trivial functions like this one are imported very reliably by ThinLTO.
    #[cfg(bpass1)]
    pub fn inlined_fn() -> u32 {
        1234
    }

    #[cfg(not(bpass1))]
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
