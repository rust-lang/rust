//! A folding traversal mechanism for complex data structures that contain type
//! information.
//!
//! This is a modifying traversal. It consumes the data structure, producing a
//! (possibly) modified version of it. Both fallible and infallible versions are
//! available. The name is potentially confusing, because this traversal is more
//! like `Iterator::map` than `Iterator::fold`.
//!
//! This traversal has limited flexibility. Only a small number of "types of
//! interest" within the complex data structures can receive custom
//! modification. These are the ones containing the most important type-related
//! information, such as `Ty`, `Predicate`, `Region`, and `Const`.
//!
//! There are three groups of traits involved in each traversal.
//! - `TypeFoldable`. This is implemented once for many types, including:
//!   - Types of interest, for which the methods delegate to the folder.
//!   - All other types, including generic containers like `Vec` and `Option`.
//!     It defines a "skeleton" of how they should be folded.
//! - `TypeSuperFoldable`. This is implemented only for each type of interest,
//!   and defines the folding "skeleton" for these types.
//! - `TypeFolder`/`FallibleTypeFolder. One of these is implemented for each
//!   folder. This defines how types of interest are folded.
//!
//! This means each fold is a mixture of (a) generic folding operations, and (b)
//! custom fold operations that are specific to the folder.
//! - The `TypeFoldable` impls handle most of the traversal, and call into
//!   `TypeFolder`/`FallibleTypeFolder` when they encounter a type of interest.
//! - A `TypeFolder`/`FallibleTypeFolder` may call into another `TypeFoldable`
//!   impl, because some of the types of interest are recursive and can contain
//!   other types of interest.
//! - A `TypeFolder`/`FallibleTypeFolder` may also call into a `TypeSuperFoldable`
//!   impl, because each folder might provide custom handling only for some types
//!   of interest, or only for some variants of each type of interest, and then
//!   use default traversal for the remaining cases.
//!
//! For example, if you have `struct S(Ty, U)` where `S: TypeFoldable` and `U:
//! TypeFoldable`, and an instance `s = S(ty, u)`, it would be folded like so:
//! ```text
//! s.fold_with(folder) calls
//! - ty.fold_with(folder) calls
//!   - folder.fold_ty(ty) may call
//!     - ty.super_fold_with(folder)
//! - u.fold_with(folder)
//! ```
use crate::{visit::TypeVisitable, Interner};

/// This trait is implemented for every type that can be folded,
/// providing the skeleton of the traversal.
///
/// To implement this conveniently, use the derive macro located in
/// `rustc_macros`.
pub trait TypeFoldable<I: Interner>: TypeVisitable<I> {
    /// The entry point for folding. To fold a value `t` with a folder `f`
    /// call: `t.try_fold_with(f)`.
    ///
    /// For most types, this just traverses the value, calling `try_fold_with`
    /// on each field/element.
    ///
    /// For types of interest (such as `Ty`), the implementation of method
    /// calls a folder method specifically for that type (such as
    /// `F::try_fold_ty`). This is where control transfers from `TypeFoldable`
    /// to `TypeFolder`.
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error>;

    /// A convenient alternative to `try_fold_with` for use with infallible
    /// folders. Do not override this method, to ensure coherence with
    /// `try_fold_with`.
    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        self.try_fold_with(folder).into_ok()
    }
}

// This trait is implemented for types of interest.
pub trait TypeSuperFoldable<I: Interner>: TypeFoldable<I> {
    /// Provides a default fold for a type of interest. This should only be
    /// called within `TypeFolder` methods, when a non-custom traversal is
    /// desired for the value of the type of interest passed to that method.
    /// For example, in `MyFolder::try_fold_ty(ty)`, it is valid to call
    /// `ty.try_super_fold_with(self)`, but any other folding should be done
    /// with `xyz.try_fold_with(self)`.
    fn try_super_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error>;

    /// A convenient alternative to `try_super_fold_with` for use with
    /// infallible folders. Do not override this method, to ensure coherence
    /// with `try_super_fold_with`.
    fn super_fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        self.try_super_fold_with(folder).into_ok()
    }
}

/// This trait is implemented for every infallible folding traversal. There is
/// a fold method defined for every type of interest. Each such method has a
/// default that does an "identity" fold. Implementations of these methods
/// often fall back to a `super_fold_with` method if the primary argument
/// doesn't satisfy a particular condition.
///
/// A blanket implementation of [`FallibleTypeFolder`] will defer to
/// the infallible methods of this trait to ensure that the two APIs
/// are coherent.
pub trait TypeFolder<I: Interner>: FallibleTypeFolder<I, Error = !> {
    fn interner(&self) -> I;

    fn fold_binder<T>(&mut self, t: I::Binder<T>) -> I::Binder<T>
    where
        T: TypeFoldable<I>,
        I::Binder<T>: TypeSuperFoldable<I>,
    {
        t.super_fold_with(self)
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty
    where
        I::Ty: TypeSuperFoldable<I>,
    {
        t.super_fold_with(self)
    }

    fn fold_region(&mut self, r: I::Region) -> I::Region
    where
        I::Region: TypeSuperFoldable<I>,
    {
        r.super_fold_with(self)
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const
    where
        I::Const: TypeSuperFoldable<I>,
    {
        c.super_fold_with(self)
    }

    fn fold_predicate(&mut self, p: I::Predicate) -> I::Predicate
    where
        I::Predicate: TypeSuperFoldable<I>,
    {
        p.super_fold_with(self)
    }
}

/// This trait is implemented for every folding traversal. There is a fold
/// method defined for every type of interest. Each such method has a default
/// that does an "identity" fold.
///
/// A blanket implementation of this trait (that defers to the relevant
/// method of [`TypeFolder`]) is provided for all infallible folders in
/// order to ensure the two APIs are coherent.
pub trait FallibleTypeFolder<I: Interner>: Sized {
    type Error;

    fn interner(&self) -> I;

    fn try_fold_binder<T>(&mut self, t: I::Binder<T>) -> Result<I::Binder<T>, Self::Error>
    where
        T: TypeFoldable<I>,
        I::Binder<T>: TypeSuperFoldable<I>,
    {
        t.try_super_fold_with(self)
    }

    fn try_fold_ty(&mut self, t: I::Ty) -> Result<I::Ty, Self::Error>
    where
        I::Ty: TypeSuperFoldable<I>,
    {
        t.try_super_fold_with(self)
    }

    fn try_fold_region(&mut self, r: I::Region) -> Result<I::Region, Self::Error>
    where
        I::Region: TypeSuperFoldable<I>,
    {
        r.try_super_fold_with(self)
    }

    fn try_fold_const(&mut self, c: I::Const) -> Result<I::Const, Self::Error>
    where
        I::Const: TypeSuperFoldable<I>,
    {
        c.try_super_fold_with(self)
    }

    fn try_fold_predicate(&mut self, p: I::Predicate) -> Result<I::Predicate, Self::Error>
    where
        I::Predicate: TypeSuperFoldable<I>,
    {
        p.try_super_fold_with(self)
    }
}

// This blanket implementation of the fallible trait for infallible folders
// delegates to infallible methods to ensure coherence.
impl<I: Interner, F> FallibleTypeFolder<I> for F
where
    F: TypeFolder<I>,
{
    type Error = !;

    fn interner(&self) -> I {
        TypeFolder::interner(self)
    }

    fn try_fold_binder<T>(&mut self, t: I::Binder<T>) -> Result<I::Binder<T>, !>
    where
        T: TypeFoldable<I>,
        I::Binder<T>: TypeSuperFoldable<I>,
    {
        Ok(self.fold_binder(t))
    }

    fn try_fold_ty(&mut self, t: I::Ty) -> Result<I::Ty, !>
    where
        I::Ty: TypeSuperFoldable<I>,
    {
        Ok(self.fold_ty(t))
    }

    fn try_fold_region(&mut self, r: I::Region) -> Result<I::Region, !>
    where
        I::Region: TypeSuperFoldable<I>,
    {
        Ok(self.fold_region(r))
    }

    fn try_fold_const(&mut self, c: I::Const) -> Result<I::Const, !>
    where
        I::Const: TypeSuperFoldable<I>,
    {
        Ok(self.fold_const(c))
    }

    fn try_fold_predicate(&mut self, p: I::Predicate) -> Result<I::Predicate, !>
    where
        I::Predicate: TypeSuperFoldable<I>,
    {
        Ok(self.fold_predicate(p))
    }
}
