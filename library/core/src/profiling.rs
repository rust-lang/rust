//! Profiling markers for compiler instrumentation.

/// Profiling marker for move operations.
///
/// This function is never called at runtime. When `-Z annotate-moves` is enabled,
/// the compiler creates synthetic debug info that makes move operations appear as
/// calls to this function in profilers.
///
/// The `SIZE` parameter encodes the size of the type being copied. It's the same as
/// `size_of::<T>()`, and is only present for convenience.
#[unstable(feature = "profiling_marker_api", issue = "148197")]
#[lang = "compiler_move"]
pub fn compiler_move<T, const SIZE: usize>(_src: *const T, _dst: *mut T) {
    unreachable!(
        "compiler_move marks where the compiler-generated a memcpy for moves. It is never actually called."
    )
}

/// Profiling marker for copy operations.
///
/// This function is never called at runtime. When `-Z annotate-moves` is enabled,
/// the compiler creates synthetic debug info that makes copy operations appear as
/// calls to this function in profilers.
///
/// The `SIZE` parameter encodes the size of the type being copied. It's the same as
/// `size_of::<T>()`, and is only present for convenience.
#[unstable(feature = "profiling_marker_api", issue = "148197")]
#[lang = "compiler_copy"]
pub fn compiler_copy<T, const SIZE: usize>(_src: *const T, _dst: *mut T) {
    unreachable!(
        "compiler_copy marks where the compiler-generated a memcpy for Copies. It is never actually called."
    )
}
