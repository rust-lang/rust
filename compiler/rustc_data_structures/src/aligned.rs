use std::ptr::Alignment;

use rustc_serialize::PointeeSized;

/// Returns the ABI-required minimum alignment of a type in bytes.
///
/// This is equivalent to [`align_of`], but also works for some unsized
/// types (e.g. slices or rustc's `List`s).
pub const fn align_of<T: ?Sized + Aligned>() -> Alignment {
    T::ALIGN
}

/// A type with a statically known alignment.
///
/// # Safety
///
/// `Self::ALIGN` must be equal to the alignment of `Self`. For sized types it
/// is [`align_of::<Self>()`], for unsized types it depends on the type, for
/// example `[T]` has alignment of `T`.
///
/// [`align_of::<Self>()`]: align_of
pub unsafe trait Aligned: PointeeSized {
    /// Alignment of `Self`.
    const ALIGN: Alignment;
}

unsafe impl<T> Aligned for T {
    const ALIGN: Alignment = Alignment::of::<Self>();
}

unsafe impl<T> Aligned for [T] {
    const ALIGN: Alignment = Alignment::of::<T>();
}
