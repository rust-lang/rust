//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Id {
    type This;
}
impl<T> Id for T {
    type This = T;
}

// We have two member constraints here:
//
// - 'unconstrained member ['a, 'static]
// - 'unconstrained member ['static]
//
// Applying the first constraint results in `'unconstrained: 'a`
// while the second then adds `'unconstrained: 'static`. If applying
// member constraints were to require the member region equal to the
// choice region, applying the first constraint first and then the
// second would result in a `'a: 'static` requirement.
fn test<'a>() -> impl Id<This = impl Sized + use<>> + use<'a> {
    &()
}
fn main() {}
