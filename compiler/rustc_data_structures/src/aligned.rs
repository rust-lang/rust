use std::mem;

/// Returns the ABI-required minimum alignment of a type in bytes.
///
/// This is equivalent to [`mem::align_of`], but also works for some unsized
/// types (e.g. slices or rustc's `List`s).
pub const fn align_of<T: ?Sized + Aligned>() -> usize {
    T::ALIGN
}

/// A type with a statically known alignment.
///
/// # Safety
///
/// `Self::ALIGN` must be equal to the alignment of `Self`. For sized types it
/// is [`mem::align_of<Self>()`], for unsized types it depends on the type, for
/// example `[T]` has alignment of `T`.
///
/// [`mem::align_of<Self>()`]: mem::align_of
pub unsafe trait Aligned {
    /// Alignment of `Self`.
    const ALIGN: usize;
}

unsafe impl<T> Aligned for T {
    const ALIGN: usize = mem::align_of::<Self>();
}

unsafe impl<T> Aligned for [T] {
    const ALIGN: usize = mem::align_of::<T>();
}
