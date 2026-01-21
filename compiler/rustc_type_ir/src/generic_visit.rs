//! A visiting traversal mechanism for complex data structures that contain type
//! information.
//!
//! This is a read-only traversal of the data structure.
//!
//! This traversal has limited flexibility. Only a small number of "types of
//! interest" within the complex data structures can receive custom
//! visitation. These are the ones containing the most important type-related
//! information, such as `Ty`, `Predicate`, `Region`, and `Const`.
//!
//! There are three traits involved in each traversal.
//! - `GenericTypeVisitable`. This is implemented once for many types, including:
//!   - Types of interest, for which the methods delegate to the visitor.
//!   - All other types, including generic containers like `Vec` and `Option`.
//!     It defines a "skeleton" of how they should be visited.
//! - `TypeSuperVisitable`. This is implemented only for recursive types of
//!   interest, and defines the visiting "skeleton" for these types. (This
//!   excludes `Region` because it is non-recursive, i.e. it never contains
//!   other types of interest.)
//! - `CustomizableTypeVisitor`. This is implemented for each visitor. This defines how
//!   types of interest are visited.
//!
//! This means each visit is a mixture of (a) generic visiting operations, and (b)
//! custom visit operations that are specific to the visitor.
//! - The `GenericTypeVisitable` impls handle most of the traversal, and call into
//!   `CustomizableTypeVisitor` when they encounter a type of interest.
//! - A `CustomizableTypeVisitor` may call into another `GenericTypeVisitable` impl, because some of
//!   the types of interest are recursive and can contain other types of interest.
//! - A `CustomizableTypeVisitor` may also call into a `TypeSuperVisitable` impl, because each
//!   visitor might provide custom handling only for some types of interest, or
//!   only for some variants of each type of interest, and then use default
//!   traversal for the remaining cases.
//!
//! For example, if you have `struct S(Ty, U)` where `S: GenericTypeVisitable` and `U:
//! GenericTypeVisitable`, and an instance `s = S(ty, u)`, it would be visited like so:
//! ```text
//! s.generic_visit_with(visitor) calls
//! - ty.generic_visit_with(visitor) calls
//!   - visitor.visit_ty(ty) may call
//!     - ty.super_generic_visit_with(visitor)
//! - u.generic_visit_with(visitor)
//! ```

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
    /// The entry point for visiting. To visit a value `t` with a visitor `v`
    /// call: `t.generic_visit_with(v)`.
    ///
    /// For most types, this just traverses the value, calling `generic_visit_with` on
    /// each field/element.
    ///
    /// For types of interest (such as `Ty`), the implementation of this method
    /// that calls a visitor method specifically for that type (such as
    /// `V::visit_ty`). This is where control transfers from `GenericTypeVisitable` to
    /// `CustomizableTypeVisitor`.
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
