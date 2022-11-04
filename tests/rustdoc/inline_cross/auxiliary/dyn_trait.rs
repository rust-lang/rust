// ignore-tidy-linelength

pub type Ty0 = dyn for<'any> FnOnce(&'any str) -> bool;

pub type Ty1<'obj> = dyn std::fmt::Display + 'obj;

pub type Ty2 = dyn for<'a, 'r> Container<'r, Item<'a, 'static> = ()>;

pub type Ty3<'s> = &'s dyn ToString;

pub trait Container<'r> {
    type Item<'a, 'ctx>;
}

// Trait-object types inside of a container type that has lifetime bounds ("wrapped").

pub fn late_bound_wrapped_elided(_: &(dyn Fn() + '_)) {}
pub fn late_bound_wrapped_late0<'f>(_: &mut (dyn Fn() + 'f)) {}
pub fn late_bound_wrapped_defaulted0<'f>(_: &'f mut dyn Fn()) {}
pub type EarlyBoundWrappedDefaulted0<'x> = std::cell::Ref<'x, dyn Trait>;
pub type EarlyBoundWrappedDefaulted1<'x> = &'x dyn Trait;
pub type EarlyBoundWrappedEarly<'x, 'y> = std::cell::Ref<'x, dyn Trait + 'y>;
pub type EarlyBoundWrappedStatic<'x> = std::cell::Ref<'x, dyn Trait + 'static>;
pub fn late_bound_wrapped_defaulted1<'l>(_: std::cell::Ref<'l, dyn Trait>) {}
pub fn late_bound_wrapped_late1<'l, 'm>(_: std::cell::Ref<'l, dyn Trait + 'm>) {}
pub fn late_bound_wrapped_early<'e, 'l>(_: std::cell::Ref<'l, dyn Trait + 'e>) where 'e: {} // `'e` is early-bound
pub fn elided_bound_wrapped_defaulted(_: std::cell::Ref<'_, dyn Trait>) {}
pub type StaticBoundWrappedDefaulted0 = std::cell::Ref<'static, dyn Trait>;
pub type StaticBoundWrappedDefaulted1 = &'static dyn Trait;
pub type AmbiguousBoundWrappedEarly0<'r, 's> = AmbiguousBoundWrapper<'s, 'r, dyn Trait + 's>;
pub type AmbiguousBoundWrappedEarly1<'r, 's> = AmbiguousBoundWrapper<'s, 'r, dyn Trait + 'r>;
pub type AmbiguousBoundWrappedStatic<'q> = AmbiguousBoundWrapper<'q, 'q, dyn Trait + 'static>;

// Trait-object types inside of a container type that doesn't have lifetime bounds ("wrapped").

pub type NoBoundsWrappedDefaulted = Box<dyn Trait>;
pub type NoBoundsWrappedEarly<'e> = Box<dyn Trait + 'e>;
pub fn no_bounds_wrapped_late<'l>(_: Box<dyn Trait + 'l>) {}
pub fn no_bounds_wrapped_elided(_: Box<dyn Trait + '_>) {}

// Trait-object types outside of a container (“bare”).

pub type BareNoBoundsDefaulted = dyn Trait;
pub type BareNoBoundsEarly<'p> = dyn Trait + 'p;
pub type BareEarlyBoundDefaulted0<'u> = dyn EarlyBoundTrait0<'u>;
pub type BareEarlyBoundDefaulted1 = dyn for<'any> EarlyBoundTrait0<'any>;
pub type BareEarlyBoundDefaulted2<'w> = dyn EarlyBoundTrait1<'static, 'w>;
pub type BareEarlyBoundEarly<'i, 'j> = dyn EarlyBoundTrait0<'i> + 'j;
pub type BareEarlyBoundStatic<'i> = dyn EarlyBoundTrait0<'i> + 'static;
pub type BareStaticBoundDefaulted = dyn StaticBoundTrait;
pub type BareHigherRankedBoundDefaulted0 = dyn HigherRankedBoundTrait0;
pub type BareHigherRankedBoundDefaulted1<'r> = dyn HigherRankedBoundTrait1<'r>;
pub type BareAmbiguousBoundEarly0<'m, 'n> = dyn AmbiguousBoundTrait<'m, 'n> + 'm;
pub type BareAmbiguousBoundEarly1<'m, 'n> = dyn AmbiguousBoundTrait<'m, 'n> + 'n;
pub type BareAmbiguousBoundStatic<'o> = dyn AmbiguousBoundTrait<'o, 'o> + 'static;

// Trait and container definitions.

pub trait Trait {} // no bounds
pub trait EarlyBoundTrait0<'b>: 'b {}
pub trait EarlyBoundTrait1<'unused, 'c>: 'c {}
pub trait StaticBoundTrait: 'static {}
pub trait HigherRankedBoundTrait0 where for<'a> Self: 'a {}
pub trait HigherRankedBoundTrait1<'e> where for<'l> Self: 'e + 'l {}
pub trait AmbiguousBoundTrait<'a, 'b>: 'a + 'b {}

pub struct AmbiguousBoundWrapper<'a, 'b, T: ?Sized + 'a + 'b>(&'a T, &'b T);
