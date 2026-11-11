//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #110534

// Regression test for #110534. This ICE'd with the old solver:
// - evaluation used the leak check in evaluation_probe
// - selection used evaluation only if there are multiple applicable candidates
// - fulfillment never used the leak check
//
// In HIR typeck, we select `Option<?x>: Trait` which has two applicable candidates,
// causing us to use evaluate_candidate. `for<'a> Option<?x>: LeakErr<'a>` of the
// first impl causes a leak check error.  `for<'a> ?x: LeakErr<'a>` of the second
// impl remains ambiguous. This causes us to select the second impl. We then only
// prove `for<'a> u32: LeakErr<'a>` in fulfillment which does not use the leak check.
// THis causes HIR typeck to pass.
//
// In MIR typeck, we select `Option<u32>: Trait` which has two applicable candidates,
// causing us to use evaluate_candidate. However, now both candidates fail the leak
// check, causing us to have no applicable impl, causing a selection error.

trait Trait {}
impl<T: for<'a> LeakErr<'a>> Trait for T {}
impl<U: for<'a> LeakErr<'a>> Trait for Option<U> {}

trait LeakErr<'a> {}
impl<T> LeakErr<'static> for T {}

fn impls_trait<T: Trait>(x: T) -> T {
    x
}

fn main() {
    let y = impls_trait(None);
    //[next]~^ ERROR: the trait bound `Option<u32>: Trait` is not satisfied
    let _: Option<u32> = y;
}
