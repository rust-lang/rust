//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] build-pass
//@[old] build-fail
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #146965
//@ compile-flags: --crate-type lib -C opt-level=3

// A regression test for #146965. When considering to inline `fn no_bound`
// into `fn optimize_me` we check `no_bound::{closure}: FnOnce(<T as Ring>::Element)`.
//
// In `fn no_bound` we can normalize `<T as Ring>::Element` to `u16` while the
// `T: Ring` bound in `fn optimize_me` means the associated type stays rigid
// again. This then causes us to fail to prove this where-bound.
//
// This is the case with both solvers, the new solver accepts it gracefully, while
// the old solver ICEd in `fn codegen_select_candidate`.

pub trait Ring {
    type Element;
}
impl<T> Ring for T {
    type Element = u16;
}

fn and_rigid_again<T: Ring>(f: impl FnOnce(<T as Ring>::Element)) {
    fn mk<T>() -> T { todo!() }
    f(mk());
}

fn no_bound<T>() {
    and_rigid_again::<T>(|_: u16| {});
}

pub fn optimize_me<T: Ring>() {
    no_bound::<T>();
}
