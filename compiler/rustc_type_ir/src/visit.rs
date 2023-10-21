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
//! - `TypeVisitable`. This is implemented once for many types, including:
//!   - Types of interest, for which the methods delegate to the visitor.
//!   - All other types, including generic containers like `Vec` and `Option`.
//!     It defines a "skeleton" of how they should be visited.
//! - `TypeSuperVisitable`. This is implemented only for recursive types of
//!   interest, and defines the visiting "skeleton" for these types. (This
//!   excludes `Region` because it is non-recursive, i.e. it never contains
//!   other types of interest.)
//! - `TypeVisitor`. This is implemented for each visitor. This defines how
//!   types of interest are visited.
//!
//! This means each visit is a mixture of (a) generic visiting operations, and (b)
//! custom visit operations that are specific to the visitor.
//! - The `TypeVisitable` impls handle most of the traversal, and call into
//!   `TypeVisitor` when they encounter a type of interest.
//! - A `TypeVisitor` may call into another `TypeVisitable` impl, because some of
//!   the types of interest are recursive and can contain other types of interest.
//! - A `TypeVisitor` may also call into a `TypeSuperVisitable` impl, because each
//!   visitor might provide custom handling only for some types of interest, or
//!   only for some variants of each type of interest, and then use default
//!   traversal for the remaining cases.
//!
//! For example, if you have `struct S(Ty, U)` where `S: TypeVisitable` and `U:
//! TypeVisitable`, and an instance `s = S(ty, u)`, it would be visited like so:
//! ```text
//! s.visit_with(visitor) calls
//! - ty.visit_with(visitor) calls
//!   - visitor.visit_ty(ty) may call
//!     - ty.super_visit_with(visitor)
//! - u.visit_with(visitor)
//! ```

use rustc_data_structures::sync::Lrc;
use rustc_index::{Idx, IndexVec};
use std::fmt;
use std::ops::ControlFlow;

use crate::Interner;

/// This trait is implemented for every type that can be visited,
/// providing the skeleton of the traversal.
///
/// To implement this conveniently, use the derive macro located in
/// `rustc_macros`.
pub trait TypeVisitable<I: Interner>: fmt::Debug + Clone {
    /// The entry point for visiting. To visit a value `t` with a visitor `v`
    /// call: `t.visit_with(v)`.
    ///
    /// For most types, this just traverses the value, calling `visit_with` on
    /// each field/element.
    ///
    /// For types of interest (such as `Ty`), the implementation of this method
    /// that calls a visitor method specifically for that type (such as
    /// `V::visit_ty`). This is where control transfers from `TypeVisitable` to
    /// `TypeVisitor`.
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy>;
}

// This trait is implemented for types of interest.
pub trait TypeSuperVisitable<I: Interner>: TypeVisitable<I> {
    /// Provides a default visit for a recursive type of interest. This should
    /// only be called within `TypeVisitor` methods, when a non-custom
    /// traversal is desired for the value of the type of interest passed to
    /// that method. For example, in `MyVisitor::visit_ty(ty)`, it is valid to
    /// call `ty.super_visit_with(self)`, but any other visiting should be done
    /// with `xyz.visit_with(self)`.
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy>;
}

/// This trait is implemented for every visiting traversal. There is a visit
/// method defined for every type of interest. Each such method has a default
/// that recurses into the type's fields in a non-custom fashion.
pub trait TypeVisitor<I: Interner>: Sized {
    type BreakTy = !;

    fn visit_binder<T: TypeVisitable<I>>(&mut self, t: &I::Binder<T>) -> ControlFlow<Self::BreakTy>
    where
        I::Binder<T>: TypeSuperVisitable<I>,
    {
        t.super_visit_with(self)
    }

    fn visit_ty(&mut self, t: I::Ty) -> ControlFlow<Self::BreakTy>
    where
        I::Ty: TypeSuperVisitable<I>,
    {
        t.super_visit_with(self)
    }

    // The default region visitor is a no-op because `Region` is non-recursive
    // and has no `super_visit_with` method to call. That also explains the
    // lack of `I::Region: TypeSuperVisitable<I>` bound.
    fn visit_region(&mut self, _r: I::Region) -> ControlFlow<Self::BreakTy> {
        ControlFlow::Continue(())
    }

    fn visit_const(&mut self, c: I::Const) -> ControlFlow<Self::BreakTy>
    where
        I::Const: TypeSuperVisitable<I>,
    {
        c.super_visit_with(self)
    }

    fn visit_predicate(&mut self, p: I::Predicate) -> ControlFlow<Self::BreakTy>
    where
        I::Predicate: TypeSuperVisitable<I>,
    {
        p.super_visit_with(self)
    }
}

///////////////////////////////////////////////////////////////////////////
// Traversal implementations.

impl<I: Interner, T: TypeVisitable<I>, U: TypeVisitable<I>> TypeVisitable<I> for (T, U) {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.0.visit_with(visitor)?;
        self.1.visit_with(visitor)
    }
}

impl<I: Interner, A: TypeVisitable<I>, B: TypeVisitable<I>, C: TypeVisitable<I>> TypeVisitable<I>
    for (A, B, C)
{
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.0.visit_with(visitor)?;
        self.1.visit_with(visitor)?;
        self.2.visit_with(visitor)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Option<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match self {
            Some(v) => v.visit_with(visitor),
            None => ControlFlow::Continue(()),
        }
    }
}

impl<I: Interner, T: TypeVisitable<I>, E: TypeVisitable<I>> TypeVisitable<I> for Result<T, E> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match self {
            Ok(v) => v.visit_with(visitor),
            Err(e) => e.visit_with(visitor),
        }
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Lrc<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Box<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Vec<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

// `TypeFoldable` isn't impl'd for `&[T]`. It doesn't make sense in the general
// case, because we can't return a new slice. But note that there are a couple
// of trivial impls of `TypeFoldable` for specific slice types elsewhere.
impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for &[T] {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Box<[T]> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<I: Interner, T: TypeVisitable<I>, Ix: Idx> TypeVisitable<I> for IndexVec<Ix, T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}
