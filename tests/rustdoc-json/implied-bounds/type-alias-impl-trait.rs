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

#[allow(dead_code)]
static STATIC_MARKER: StaticMarker = StaticMarker;

//@ has "$.index[?(@.name=='Opaque')].inner.type_alias.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='Opaque')].inner.type_alias.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='Opaque')].inner.type_alias.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Debug')]"
pub type Opaque = impl Debug;

//@ has "$.index[?(@.name=='OpaqueRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='OpaqueRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='OpaqueRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='OpaqueRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
pub type OpaqueRef = &'static impl Debug;

//@ has "$.index[?(@.name=='OpaqueMaybeUnsized')].inner.type_alias.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='OpaqueMaybeUnsized')].inner.type_alias.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ count "$.index[?(@.name=='OpaqueMaybeUnsized')].inner.type_alias.type.impl_trait.implied_bounds[*]" 0
pub type OpaqueMaybeUnsized = impl Debug + ?Sized;

//@ has "$.index[?(@.name=='OpaqueMaybeUnsizedRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='OpaqueMaybeUnsizedRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='OpaqueMaybeUnsizedRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ !has "$.index[?(@.name=='OpaqueMaybeUnsizedRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ has "$.index[?(@.name=='OpaqueMaybeUnsizedRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
pub type OpaqueMaybeUnsizedRef = &'static (impl Debug + ?Sized);

//@ has "$.index[?(@.name=='OpaqueSizedViaTrait')].inner.type_alias.type.impl_trait.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='OpaqueSizedViaTrait')].inner.type_alias.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='OpaqueSizedViaTrait')].inner.type_alias.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
pub type OpaqueSizedViaTrait = impl NeedsSized;

//@ has "$.index[?(@.name=='OpaqueSizedViaTraitRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='OpaqueSizedViaTraitRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='OpaqueSizedViaTraitRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='OpaqueSizedViaTraitRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
pub type OpaqueSizedViaTraitRef = &'static impl NeedsSized;

//@ has "$.index[?(@.name=='OpaqueOverridden')].inner.type_alias.type.impl_trait.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='OpaqueOverridden')].inner.type_alias.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='OpaqueOverridden')].inner.type_alias.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='OpaqueOverridden')].inner.type_alias.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
pub type OpaqueOverridden = impl NeedsSized + ?Sized;

//@ has "$.index[?(@.name=='OpaqueOverriddenRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='OpaqueOverriddenRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='OpaqueOverriddenRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='OpaqueOverriddenRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='OpaqueOverriddenRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
pub type OpaqueOverriddenRef = &'static (impl NeedsSized + ?Sized);

//@ has "$.index[?(@.name=='OpaqueStatic')].inner.type_alias.type.impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has "$.index[?(@.name=='OpaqueStatic')].inner.type_alias.type.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
//@ has "$.index[?(@.name=='OpaqueStatic')].inner.type_alias.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='OpaqueStatic')].inner.type_alias.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
pub type OpaqueStatic = impl StaticOnly;

//@ has "$.index[?(@.name=='OpaqueStaticRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has "$.index[?(@.name=='OpaqueStaticRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ has "$.index[?(@.name=='OpaqueStaticRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='OpaqueStaticRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
pub type OpaqueStaticRef = &'static impl StaticOnly;

//@ has "$.index[?(@.name=='OpaqueStaticMaybeUnsized')].inner.type_alias.type.impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has "$.index[?(@.name=='OpaqueStaticMaybeUnsized')].inner.type_alias.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='OpaqueStaticMaybeUnsized')].inner.type_alias.type.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='OpaqueStaticMaybeUnsized')].inner.type_alias.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
pub type OpaqueStaticMaybeUnsized = impl StaticOnly + ?Sized;

//@ has "$.index[?(@.name=='OpaqueStaticMaybeUnsizedRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has "$.index[?(@.name=='OpaqueStaticMaybeUnsizedRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='OpaqueStaticMaybeUnsizedRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='OpaqueStaticMaybeUnsizedRef')].inner.type_alias.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
pub type OpaqueStaticMaybeUnsizedRef = &'static (impl StaticOnly + ?Sized);

// Ensure the opaques we defined above are not unconstrained.
#[allow(dead_code)]
#[define_opaque(
    Opaque,
    OpaqueRef,
    OpaqueMaybeUnsized,
    OpaqueMaybeUnsizedRef,
    OpaqueSizedViaTrait,
    OpaqueSizedViaTraitRef,
    OpaqueOverridden,
    OpaqueOverriddenRef,
    OpaqueStatic,
    OpaqueStaticRef,
    OpaqueStaticMaybeUnsized,
    OpaqueStaticMaybeUnsizedRef
)]
fn define_all() -> (
    Opaque,
    OpaqueRef,
    Box<OpaqueMaybeUnsized>,
    OpaqueMaybeUnsizedRef,
    OpaqueSizedViaTrait,
    OpaqueSizedViaTraitRef,
    OpaqueOverridden,
    OpaqueOverriddenRef,
    OpaqueStatic,
    OpaqueStaticRef,
    Box<OpaqueStaticMaybeUnsized>,
    OpaqueStaticMaybeUnsizedRef,
) {
    (
        StaticMarker,
        &STATIC_MARKER,
        Box::new(StaticMarker),
        &STATIC_MARKER,
        StaticMarker,
        &STATIC_MARKER,
        StaticMarker,
        &STATIC_MARKER,
        StaticMarker,
        &STATIC_MARKER,
        Box::new(StaticMarker),
        &STATIC_MARKER,
    )
}
