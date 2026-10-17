//! Platform-specific extensions to `core` for Windows platforms.

#![unstable(feature = "codeview_annotation", issue = "163964")]
#![doc(cfg(windows))]

#[doc(inline)]
pub use crate::intrinsics::CodeViewAnnotationArgs;

/// Writes [`T::ARGS`](CodeViewAnnotationArgs::ARGS) to the PDB as an `S_ANNOTATION` record using
/// [`llvm.codeview.annotation`](https://llvm.org/docs/LangRef.html#llvm-codeview-annotation-intrinsic).
///
/// This function has an effect only on the `msvc` environment and the LLVM backend. It is a no-op on other
/// environments and backends.
///
/// # Examples
///
/// To call `codeview_annotation`, the caller must declare a type
/// implementing [`CodeViewAnnotationArgs`] and pass it as the type
/// parameter to `codeview_annotation`. The string arguments must be
/// specified in `CodeViewAnnotationArgs::ARGS`.
///
#[cfg_attr(windows, doc = "```")]
#[cfg_attr(not(windows), doc = "```ignore (needs windows)")]
/// #![feature(codeview_annotation, core_os)]
/// use core::os::windows::{codeview_annotation, CodeViewAnnotationArgs};
///
/// struct Args;
///
/// impl CodeViewAnnotationArgs for Args {
///     const ARGS: &[&str] = &["Hello", "World"];
/// }
///
/// codeview_annotation::<Args>();
/// ```
#[inline(always)]
#[unstable(feature = "codeview_annotation", issue = "163964")]
pub fn codeview_annotation<T: CodeViewAnnotationArgs>() {
    crate::intrinsics::codeview_annotation::<T>();
}
