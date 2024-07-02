// FIXME(fmease): Should we allow this as part of this MVP?
// Of course, under #22519 (arbitrary shorthand projections), this should obviously typeck.
// For reference, `T::Alias` typeck'ing does *not* imply `Identity<T>::Alias` typeck'ing.
//@ check-pass

type Identity<T> = T;

trait Trait {
    type Project: Trait;
}

fn scope<T: Trait>() {
    let _: Identity<T::Project>::Project;
}

fn main() {}
