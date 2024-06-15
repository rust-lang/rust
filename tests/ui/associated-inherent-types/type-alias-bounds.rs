//@ compile-flags: --crate-type=lib
//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// FIXME(inherent_associated_types):
// While we currently do take some clauses of the ParamEnv into consideration
// when performing IAT selection, we do not perform full well-formedness checking
// for (eager) type alias definition and usage sites.
//
// Therefore it's *correct* for lint `type_alias_bounds` to fire here despite the
// fact that removing `Bound` from `T` in `Alias` would lead to an error!
//
// Obviously, the present situation isn't ideal and we should fix it in one way
// or another. Either we somehow delay IAT selection until after HIR ty lowering
// to avoid the need to specify any bounds inside (eager) type aliases or we
// force the overarching type alias to be *lazy* (similar to TAITs) which would
// automatically lead to full wfchecking and lint TAB getting suppressed.

pub type Alias<T: Bound> = (Source<T>::Assoc,);
//~^ WARN bounds on generic parameters in type aliases are not enforced

pub struct Source<T>(T);
pub trait Bound {}

impl<T: Bound> Source<T> {
    pub type Assoc = ();
}
