// FIXME(fmease): Proper explainer.
// Of course, under #22519 (arbitrary shorthand projections), this should obviously typeck.
// For reference, `T::Alias` typeck'ing does *not* imply `Identity<T>::Alias` typeck'ing.
//@ revisions: eager lazy
//@[eager] check-pass
#![cfg_attr(lazy, feature(lazy_type_alias), allow(incomplete_features))]

type Identity<T> = T;

trait Trait {
    type Project: Trait;
}

fn scope<T: Trait>() {
    // FIXME(fmease): Reject this under [eager], too!
    let _: Identity<T::Project>::Project; //[lazy]~ ERROR ambiguous associated type
}

fn main() {}
