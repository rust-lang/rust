//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] known-bug: #141124
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr

// This example failed with the old solver in `fn rematch_impl`,
// likely due to a bug with the projection cache. Regression test
// for #141124.

struct S;
trait SimpleTrait {}
trait TraitAssoc {
    type Assoc;
}

impl<T> TraitAssoc for T
where
    T: SimpleTrait,
{
    type Assoc = <(T,) as TraitAssoc>::Assoc;
    //[next]~^ ERROR: overflow evaluating the requirement `<(T,) as TraitAssoc>::Assoc == _`
    //[next]~| ERROR: overflow evaluating whether `<(T,) as TraitAssoc>::Assoc` is well-formed
    //[next]~| ERROR: overflow evaluating the requirement `<T as TraitAssoc>::Assoc == _`
}
impl SimpleTrait for <S as TraitAssoc>::Assoc {}
//[next]~^ ERROR: overflow evaluating the requirement `<S as TraitAssoc>::Assoc == _`
//[next]~| ERROR: overflow evaluating the requirement `<S as TraitAssoc>::Assoc: SimpleTrait`
//[next]~| ERROR: overflow evaluating whether `<S as TraitAssoc>::Assoc` is well-formed

pub fn main() {}
