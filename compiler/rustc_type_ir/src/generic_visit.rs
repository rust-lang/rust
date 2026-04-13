//! Special visiting used by rust-analyzer only.
//!
//! It is different from `TypeVisitable` in two ways:
//!
//!  - The visitor is a generic of the trait and not the method, allowing types to attach
//!    special behavior to visitors (as long as they know it; we don't use this capability
//!    in rustc crates, but rust-analyzer needs it).
//!  - It **must visit** every field. This is why we don't have an attribute like `#[type_visitable(ignore)]`
//!    for this visit. The reason for this is soundness: rust-analyzer uses this visit to
//!    garbage collect types, so a missing field can mean a use after free

use std::sync::Arc;

use rustc_index::{Idx, IndexVec};
use smallvec::SmallVec;
use thin_vec::ThinVec;

/// This trait is implemented for every type that can be visited,
/// providing the skeleton of the traversal.
///
/// To implement this conveniently, use the derive macro located in
/// `rustc_macros`.
pub trait GenericTypeVisitable<V> {
    fn generic_visit_with(&self, visitor: &mut V);
}

///////////////////////////////////////////////////////////////////////////
// Traversal implementations.

impl<V, T: ?Sized + GenericTypeVisitable<V>> GenericTypeVisitable<V> for &T {
    fn generic_visit_with(&self, visitor: &mut V) {
        T::generic_visit_with(*self, visitor)
    }
}

impl<V, T: GenericTypeVisitable<V>, U: GenericTypeVisitable<V>> GenericTypeVisitable<V> for (T, U) {
    fn generic_visit_with(&self, visitor: &mut V) {
        self.0.generic_visit_with(visitor);
        self.1.generic_visit_with(visitor);
    }
}

impl<V, A: GenericTypeVisitable<V>, B: GenericTypeVisitable<V>, C: GenericTypeVisitable<V>>
    GenericTypeVisitable<V> for (A, B, C)
{
    fn generic_visit_with(&self, visitor: &mut V) {
        self.0.generic_visit_with(visitor);
        self.1.generic_visit_with(visitor);
        self.2.generic_visit_with(visitor);
    }
}

impl<V, T: GenericTypeVisitable<V>> GenericTypeVisitable<V> for Option<T> {
    fn generic_visit_with(&self, visitor: &mut V) {
        match self {
            Some(v) => v.generic_visit_with(visitor),
            None => {}
        }
    }
}

impl<V, T: GenericTypeVisitable<V>, E: GenericTypeVisitable<V>> GenericTypeVisitable<V>
    for Result<T, E>
{
    fn generic_visit_with(&self, visitor: &mut V) {
        match self {
            Ok(v) => v.generic_visit_with(visitor),
            Err(e) => e.generic_visit_with(visitor),
        }
    }
}

impl<V, T: ?Sized + GenericTypeVisitable<V>> GenericTypeVisitable<V> for Arc<T> {
    fn generic_visit_with(&self, visitor: &mut V) {
        (**self).generic_visit_with(visitor)
    }
}

impl<V, T: ?Sized + GenericTypeVisitable<V>> GenericTypeVisitable<V> for Box<T> {
    fn generic_visit_with(&self, visitor: &mut V) {
        (**self).generic_visit_with(visitor)
    }
}

impl<V, T: GenericTypeVisitable<V>> GenericTypeVisitable<V> for Vec<T> {
    fn generic_visit_with(&self, visitor: &mut V) {
        self.iter().for_each(|it| it.generic_visit_with(visitor));
    }
}

impl<V, T: GenericTypeVisitable<V>> GenericTypeVisitable<V> for ThinVec<T> {
    fn generic_visit_with(&self, visitor: &mut V) {
        self.iter().for_each(|it| it.generic_visit_with(visitor));
    }
}

impl<V, T: GenericTypeVisitable<V>, const N: usize> GenericTypeVisitable<V> for SmallVec<[T; N]> {
    fn generic_visit_with(&self, visitor: &mut V) {
        self.iter().for_each(|it| it.generic_visit_with(visitor));
    }
}

impl<V, T: GenericTypeVisitable<V>> GenericTypeVisitable<V> for [T] {
    fn generic_visit_with(&self, visitor: &mut V) {
        self.iter().for_each(|it| it.generic_visit_with(visitor));
    }
}

impl<V, T: GenericTypeVisitable<V>, Ix: Idx> GenericTypeVisitable<V> for IndexVec<Ix, T> {
    fn generic_visit_with(&self, visitor: &mut V) {
        self.iter().for_each(|it| it.generic_visit_with(visitor));
    }
}

impl<S, V> GenericTypeVisitable<V> for std::hash::BuildHasherDefault<S> {
    fn generic_visit_with(&self, _visitor: &mut V) {}
}

#[expect(rustc::default_hash_types, rustc::potential_query_instability)]
impl<
    Visitor,
    Key: GenericTypeVisitable<Visitor>,
    Value: GenericTypeVisitable<Visitor>,
    S: GenericTypeVisitable<Visitor>,
> GenericTypeVisitable<Visitor> for std::collections::HashMap<Key, Value, S>
{
    fn generic_visit_with(&self, visitor: &mut Visitor) {
        self.iter().for_each(|it| it.generic_visit_with(visitor));
        self.hasher().generic_visit_with(visitor);
    }
}

#[expect(rustc::default_hash_types, rustc::potential_query_instability)]
impl<V, T: GenericTypeVisitable<V>, S: GenericTypeVisitable<V>> GenericTypeVisitable<V>
    for std::collections::HashSet<T, S>
{
    fn generic_visit_with(&self, visitor: &mut V) {
        self.iter().for_each(|it| it.generic_visit_with(visitor));
        self.hasher().generic_visit_with(visitor);
    }
}

impl<
    Visitor,
    Key: GenericTypeVisitable<Visitor>,
    Value: GenericTypeVisitable<Visitor>,
    S: GenericTypeVisitable<Visitor>,
> GenericTypeVisitable<Visitor> for indexmap::IndexMap<Key, Value, S>
{
    fn generic_visit_with(&self, visitor: &mut Visitor) {
        self.iter().for_each(|it| it.generic_visit_with(visitor));
        self.hasher().generic_visit_with(visitor);
    }
}

impl<V, T: GenericTypeVisitable<V>, S: GenericTypeVisitable<V>> GenericTypeVisitable<V>
    for indexmap::IndexSet<T, S>
{
    fn generic_visit_with(&self, visitor: &mut V) {
        self.iter().for_each(|it| it.generic_visit_with(visitor));
        self.hasher().generic_visit_with(visitor);
    }
}

macro_rules! trivial_impls {
    ( $($ty:ty),* $(,)? ) => {
        $(
            impl<V>
                GenericTypeVisitable<V> for $ty
            {
                fn generic_visit_with(&self, _visitor: &mut V) {}
            }
        )*
    };
}

impl<T: ?Sized, V> GenericTypeVisitable<V> for std::marker::PhantomData<T> {
    fn generic_visit_with(&self, _visitor: &mut V) {}
}

trivial_impls!(
    (),
    rustc_ast_ir::Mutability,
    bool,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    crate::PredicatePolarity,
    crate::BoundConstness,
    crate::AliasRelationDirection,
    crate::DebruijnIndex,
    crate::solve::Certainty,
    crate::UniverseIndex,
    crate::BoundVar,
    crate::InferTy,
    crate::IntTy,
    crate::UintTy,
    crate::FloatTy,
    crate::InferConst,
    crate::RegionVid,
    rustc_hash::FxBuildHasher,
    crate::TypeFlags,
    crate::solve::GoalSource,
);
