//! To improve compile times and code size for the compiler itself, query
//! values are "erased" in some contexts (e.g. inside in-memory cache types),
//! to reduce the number of generic instantiations created during codegen.

use std::intrinsics::transmute_unchecked;
use std::mem::MaybeUninit;

/// Internal implementation detail of [`Erased`].
#[derive(Copy, Clone)]
pub struct ErasedData<Storage: Copy> {
    /// We use `MaybeUninit` here to make sure it's legal to store a transmuted
    /// value that isn't actually of type `Storage`.
    data: MaybeUninit<Storage>,
}

/// Trait for types that can be erased into [`Erased<Self>`].
///
/// Erasing and unerasing values is performed by [`erase_val`] and [`restore_val`].
pub trait Erasable: Copy {
    /// Storage type to used for erased values of this type.
    /// Should be `[u8; N]`, where N is equal to `size_of::<Self>`.
    ///
    /// [`ErasedData`] wraps this storage type in `MaybeUninit` to ensure that
    /// transmutes to/from erased storage are well-defined.
    type Storage: Copy;
}

/// A value of `T` that has been "erased" into some opaque storage type.
///
/// This is helpful for reducing the number of concrete instantiations needed
/// during codegen when building the compiler.
///
/// Using an opaque type alias allows the type checker to enforce that
/// `Erased<T>` and `Erased<U>` are still distinct types, while allowing
/// monomorphization to see that they might actually use the same storage type.
pub type Erased<T: Erasable> = ErasedData<impl Copy>;

/// Erases a value of type `T` into `Erased<T>`.
///
/// `Erased<T>` and `Erased<U>` are type-checked as distinct types, but codegen
/// can see whether they actually have the same storage type.
#[inline(always)]
#[define_opaque(Erased)]
pub fn erase_val<T: Erasable>(value: T) -> Erased<T> {
    // Ensure the sizes match
    const {
        if size_of::<T>() != size_of::<T::Storage>() {
            panic!("size of T must match erased type <T as Erasable>::Storage")
        }
    };

    ErasedData::<<T as Erasable>::Storage> {
        // `transmute_unchecked` is needed here because it does not have `transmute`'s size check
        // (and thus allows to transmute between `T` and `MaybeUninit<T::Storage>`) (we do the size
        // check ourselves in the `const` block above).
        //
        // `transmute_copy` is also commonly used for this (and it would work here since
        // `Erasable: Copy`), but `transmute_unchecked` better explains the intent.
        //
        // SAFETY: It is safe to transmute to MaybeUninit for types with the same sizes.
        data: unsafe { transmute_unchecked::<T, MaybeUninit<T::Storage>>(value) },
    }
}

/// Restores an erased value to its real type.
///
/// This relies on the fact that `Erased<T>` and `Erased<U>` are type-checked
/// as distinct types, even if they use the same storage type.
#[inline(always)]
#[define_opaque(Erased)]
pub fn restore_val<T: Erasable>(erased_value: Erased<T>) -> T {
    let ErasedData { data }: ErasedData<<T as Erasable>::Storage> = erased_value;
    // See comment in `erase` for why we use `transmute_unchecked`.
    //
    // SAFETY: Due to the use of impl Trait in `Erase` the only way to safely create an instance
    // of `Erase` is to call `erase`, so we know that `value.data` is a valid instance of `T` of
    // the right size.
    unsafe { transmute_unchecked::<MaybeUninit<T::Storage>, T>(data) }
}

impl<T: Copy> Erasable for T {
    type Storage = T;
}
