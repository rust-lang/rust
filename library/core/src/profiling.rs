//! Profiling markers for compiler instrumentation.

/// Profiling marker for move operations.
///
/// This function is never called at runtime. When `-Z annotate-moves` is enabled,
/// the compiler creates synthetic debug info that makes move operations appear as
/// calls to this function in profilers.
///
/// The `SIZE` parameter encodes the size of the type being moved.
#[unstable(feature = "profiling_marker_api", issue = "none")]
#[rustc_force_inline]
#[rustc_diagnostic_item = "compiler_move"]
pub fn compiler_move<T, const SIZE: usize>(_src: *const T, _dst: *mut T) {
    unreachable!("compiler_move should never be called - it's only for debug info")
}

/// Profiling marker for copy operations.
///
/// This function is never called at runtime. When `-Z annotate-moves` is enabled,
/// the compiler creates synthetic debug info that makes copy operations appear as
/// calls to this function in profilers.
///
/// The `SIZE` parameter encodes the size of the type being copied.
#[unstable(feature = "profiling_marker_api", issue = "none")]
#[rustc_force_inline]
#[rustc_diagnostic_item = "compiler_copy"]
pub fn compiler_copy<T, const SIZE: usize>(_src: *const T, _dst: *mut T) {
    unreachable!("compiler_copy should never be called - it's only for debug info")
}
