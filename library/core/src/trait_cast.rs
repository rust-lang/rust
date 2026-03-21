//! Trait casting support.

use crate::marker::{MetaSized, TraitMetadataTable};
use crate::ptr::{DynMetadata, NonNull, Pointee};

/// Represents errors that may occur when attempting to perform a cast
/// between trait objects within the same trait graph.
///
/// This enum is generic over type `T`, which is intended to represent
/// the object involved in the cast operation.
///
/// # Variants
///
/// - `ForeignTraitGraph(T)`:
///   Indicates that the object being cast originates from a different
///   global crate than the one attempting the cast.
///   - This is useful to provide more context when debugging cast errors.
///   - **Note:** Do not rely on this behavior, as it is subject to change
///     in future versions.
///
/// - `UnsatisfiedObligation(T)`:
///   Indicates that the object being cast does not implement the required
///   trait or the cast fails due to inability to satisfy lifetime erasure
///   requirements.
///   - This may occur when the cast violates safety or does not align with
///     the constraints of the target trait.
///
/// # Usage
///
/// This enum is primarily used to encapsulate errors in trait casting
/// scenarios where such operations require validation of compatibility
/// at runtime.
#[derive(Debug, Clone, Copy)]
pub enum TraitCastError<T> {
    /// This object is from a different global crate than the one
    /// that is performing the cast.
    /// Useful if you'd like to provide a more informative error message.
    /// Note: do not rely on this behavior. It is subject to change.
    ForeignTraitGraph(T),
    /// This object does not implement the specified trait, or the cast does not
    /// satisfy lifetime erasure requirements.
    UnsatisfiedObligation(T),
}
impl<T> TraitCastError<T> {
    /// Unwrap the contained, un-casted, value.
    pub fn unwrap(self) -> T {
        match self {
            Self::ForeignTraitGraph(v) | Self::UnsatisfiedObligation(v) => v,
        }
    }
}

/// `I` is the root supertrait, `U` is the target trait.
///
/// The choice of root supertrait does not affect the value of the cast:
/// the output vtable is the same after monomorphization (or is
/// essentially user-invisible).
pub trait TraitCast<I: MetaSized, U: MetaSized>: Sized
where
    I: Pointee<Metadata = DynMetadata<I>> + TraitMetadataTable<I>,
    U: Pointee<Metadata = DynMetadata<U>> + TraitMetadataTable<I>,
{
    /// The target *value* of a successful cast.
    type Target;
    /// Attempt to cast `self` to `U`. All trait impl-obligations are enforced,
    /// but lifetime-erasure soundness is not.
    /// Returns Err(TraitCastError::UnsatisfiedObligation) if the cast is not
    /// possible due to unfulfilled generic obligations.
    /// Returns Err(TraitCastError::ForeignTraitGraph) if the cast is not
    /// possible because the object is from a different global crate.
    unsafe fn unchecked_cast(self) -> Result<Self::Target, TraitCastError<Self>>;
    /// Attempt to cast `self` to `U`.
    ///
    /// Returns Err(TraitCastError::ForeignTraitGraph) if the cast is not
    /// possible because the object is from a different global crate.
    /// Returns Err(TraitCastError::UnsatisfiedObligation) if the cast is not
    /// possible due to lifetime erasure requirements or because of unfulfilled
    /// generic obligations.
    fn checked_cast(self) -> Result<Self::Target, TraitCastError<Self>> {
        // SAFETY: `unchecked_cast`'s only precondition is that lifetime-erasure
        // soundness has been verified, since it enforces trait impl-obligations
        // on its own. The `trait_cast_is_lifetime_erasure_safe::<I, U>`
        // intrinsic call below returns `false` whenever that requirement is
        // not satisfied, and we bail out with `UnsatisfiedObligation` in that
        // case — so by the time we reach `self.unchecked_cast()`, erasure
        // soundness for `I -> U` has been established.
        unsafe {
            if !crate::intrinsics::trait_cast_is_lifetime_erasure_safe::<I, U>() {
                return Err(TraitCastError::UnsatisfiedObligation(self));
            }
            self.unchecked_cast()
        }
    }
    /// Same as `checked_cast`, but strips TraitCastError::* from the return type.
    fn cast(self) -> Result<Self::Target, Self> {
        self.checked_cast().map_err(TraitCastError::unwrap)
    }
}
impl<'r, T, U, I> TraitCast<I, U> for &'r T
where
    I: MetaSized + Pointee<Metadata = DynMetadata<I>> + TraitMetadataTable<I> + 'r,
    T: MetaSized + TraitMetadataTable<I>,
    U: MetaSized + Pointee<Metadata = DynMetadata<U>> + TraitMetadataTable<I> + 'r,
{
    type Target = &'r U;
    unsafe fn unchecked_cast(self) -> Result<&'r U, TraitCastError<Self>> {
        // SAFETY: the caller has promised lifetime-erasure soundness for
        // `I -> U` (the sole precondition of `unchecked_cast`). The compiler
        // emits `trait_metadata_index` and `trait_metadata_table` in lockstep,
        // so once `crate_graph_id == obj_graph_id` we know `table` has length
        // `trait_metadata_table_len::<I>()` and `idx` is a valid slot in it. A
        // `Some(vtable)` entry holds the `U`-vtable for `I`'s trait graph,
        // which has the layout of `DynMetadata<U>`; the transmute reconstructs
        // the correct metadata. The resulting fat pointer shares `self`'s
        // provenance and is valid for `'r` because `T: 'r` and `U: 'r`.
        unsafe {
            let (obj_graph_id, table) = <T as TraitMetadataTable<I>>::derived_metadata_table(self);
            let (crate_graph_id, idx) = crate::intrinsics::trait_metadata_index::<I, U>();
            if crate_graph_id as *const u8 != obj_graph_id as *const u8 {
                return Err(TraitCastError::ForeignTraitGraph(self));
            }

            let table_len = crate::intrinsics::trait_metadata_table_len::<I>();
            let table: &[Option<NonNull<()>>] =
                &*crate::ptr::from_raw_parts(table.as_ptr(), table_len);

            let (p, _) = (self as *const T).to_raw_parts();
            let Some(&Some(vtable)) = table.get(idx) else {
                return Err(TraitCastError::UnsatisfiedObligation(self));
            };
            Ok(&*crate::ptr::from_raw_parts(p, crate::mem::transmute(vtable)))
        }
    }
}

impl<'r, T, U, I> TraitCast<I, U> for &'r mut T
where
    I: MetaSized + Pointee<Metadata = DynMetadata<I>> + TraitMetadataTable<I> + 'r,
    T: MetaSized + TraitMetadataTable<I>,
    U: MetaSized + Pointee<Metadata = DynMetadata<U>> + TraitMetadataTable<I> + 'r,
{
    type Target = &'r mut U;
    unsafe fn unchecked_cast(self) -> Result<&'r mut U, TraitCastError<Self>> {
        // SAFETY: the caller has promised lifetime-erasure soundness for
        // `I -> U` (the sole precondition of `unchecked_cast`). The compiler
        // emits `trait_metadata_index` and `trait_metadata_table` in lockstep,
        // so once `crate_graph_id == obj_graph_id` we know `table` has length
        // `trait_metadata_table_len::<I>()` and `idx` is a valid slot in it. A
        // `Some(vtable)` entry holds the `U`-vtable for `I`'s trait graph,
        // which has the layout of `DynMetadata<U>`; the transmute reconstructs
        // the correct metadata. Uniqueness is preserved because `self` is
        // moved into this call, and the resulting `&'r mut U` shares `self`'s
        // provenance and is valid for `'r` because `T: 'r` and `U: 'r`.
        unsafe {
            let (obj_graph_id, table) = <T as TraitMetadataTable<I>>::derived_metadata_table(self);
            let (crate_graph_id, idx) = crate::intrinsics::trait_metadata_index::<I, U>();
            if crate_graph_id as *const u8 != obj_graph_id as *const u8 {
                return Err(TraitCastError::ForeignTraitGraph(self));
            }

            let table_len = crate::intrinsics::trait_metadata_table_len::<I>();
            let table: &[Option<NonNull<()>>] =
                &*crate::ptr::from_raw_parts(table.as_ptr(), table_len);

            let (p, _) = (self as *mut T).to_raw_parts();
            let Some(&Some(vtable)) = table.get(idx) else {
                return Err(TraitCastError::UnsatisfiedObligation(self));
            };
            Ok(&mut *crate::ptr::from_raw_parts_mut(p, crate::mem::transmute(vtable)))
        }
    }
}

/// Attempt to cast `$e` to `$u` in the trait graph of `$i`.
/// Returns Err($e) if the cast is not possible.
#[macro_export]
macro_rules! cast {
    (in $i:ty, $e:expr => $u:ty) => {{ $crate::trait_cast::TraitCast::<$i, $u>::cast($e) }};
}

/// Attempt to cast `$e` to `$u` in the trait graph of `$i`.
///
/// Returns Err(TraitCastError::ForeignTraitGraph) if the cast is not
/// possible because the object is from a different global crate.
/// Returns Err(TraitCastError::UnsatisfiedObligation) if the cast is not
/// possible due to lifetime erasure requirements or because of unfulfilled
/// generic obligations.
#[macro_export]
macro_rules! try_cast {
    (in $i:ty, $e:expr => $u:ty) => {{ $crate::trait_cast::TraitCast::<$i, $u>::checked_cast($e) }};
}

/// Unsafely attempt to cast `$e` to `$u` in the trait graph of `$i`.
///
/// All trait impl-obligations are enforced, but lifetime-erasure soundness is
/// not.
///
/// Returns Err(TraitCastError::UnsatisfiedObligation) if the cast is not
/// possible due to unfulfilled generic obligations.
/// Returns Err(TraitCastError::ForeignTraitGraph) if the cast is not
/// possible because the object is from a different global crate.
#[macro_export]
macro_rules! unchecked_cast {
    (in $i:ty, $e:expr => $u:ty) => {{ $crate::trait_cast::TraitCast::<$i, $u>::unchecked_cast($e) }};
}
