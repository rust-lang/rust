#![feature(type_alias_impl_trait)]

use std::fmt::{self, Debug};

pub trait NeedsSized: Sized {}
impl<T: Sized> NeedsSized for T {}

pub trait StaticOnly: 'static {}
impl<T: 'static> StaticOnly for T {}

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct StaticMarker;

impl fmt::Debug for StaticMarker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StaticMarker")
    }
}

impl<T: ?Sized> PartialEq<T> for StaticMarker {
    fn eq(&self, _: &T) -> bool {
        true
    }
}

//@ has "$.index[?(@.name=='SizedParam')].inner.type_alias.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='SizedParam')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='SizedParam')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
pub type SizedParam<'a, T: NeedsSized> = impl Debug + PartialEq<T> + 'a;

#[allow(dead_code)]
#[define_opaque(SizedParam)]
fn define_sized_param<'a, T: NeedsSized>() -> SizedParam<'a, T> {
    StaticMarker
}

//@ has "$.index[?(@.name=='MaybeSizedParam')].inner.type_alias.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='MaybeSizedParam')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='MaybeSizedParam')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
pub type MaybeSizedParam<'a, T: ?Sized> = impl Debug + PartialEq<T> + 'a;

#[allow(dead_code)]
#[define_opaque(MaybeSizedParam)]
fn define_maybe_sized_param<'a, T: ?Sized>() -> MaybeSizedParam<'a, T> {
    StaticMarker
}

//@ has "$.index[?(@.name=='SizedMaybeUnsized')].inner.type_alias.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='SizedMaybeUnsized')].inner.type_alias.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='SizedMaybeUnsized')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='SizedMaybeUnsized')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub type SizedMaybeUnsized<'a, T: NeedsSized + ?Sized> = impl Debug + PartialEq<T> + 'a;

#[allow(dead_code)]
#[define_opaque(SizedMaybeUnsized)]
fn define_sized_maybe_unsized<'a, T: NeedsSized + ?Sized>() -> SizedMaybeUnsized<'a, T> {
    StaticMarker
}

//@ has "$.index[?(@.name=='StaticParam')].inner.type_alias.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has "$.index[?(@.name=='StaticParam')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ has "$.index[?(@.name=='StaticParam')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='StaticParam')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
pub type StaticParam<'a, T: StaticOnly> = impl Debug + PartialEq<T> + 'a;

#[allow(dead_code)]
#[define_opaque(StaticParam)]
fn define_static_param<'a, T: StaticOnly>() -> StaticParam<'a, T> {
    StaticMarker
}

//@ has "$.index[?(@.name=='StaticMaybeUnsized')].inner.type_alias.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has "$.index[?(@.name=='StaticMaybeUnsized')].inner.type_alias.generics.params[1].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='StaticMaybeUnsized')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='StaticMaybeUnsized')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ !has "$.index[?(@.name=='StaticMaybeUnsized')].inner.type_alias.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub type StaticMaybeUnsized<'a, T: StaticOnly + ?Sized> = impl Debug + PartialEq<T> + 'a;

#[allow(dead_code)]
#[define_opaque(StaticMaybeUnsized)]
fn define_static_maybe_unsized<'a, T: StaticOnly + ?Sized>() -> StaticMaybeUnsized<'a, T> {
    StaticMarker
}
