//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// https://github.com/rust-lang/rust/issues/56806
//
// This results in a query cycle with the new solver. Let's not
// bother with this when stabilizing the new solver. For more context,
// see https://rust-lang.zulipchat.com/#narrow/channel/364551-t-types.2Ftrait-system-refactor/topic/dyn.20compatibility.20check.20in.20object.20candidate.20causes.20cycle/with/627370998

pub trait Trait {
    fn dyn_instead_of_self(self: Box<dyn Trait>);
    //[old]~^ ERROR: invalid `self` parameter type
    //[next]~^^^ ERROR: cycle detected when checking if trait `Trait` is dyn-compatible
}

pub fn main() {}
