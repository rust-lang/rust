//! Platform-specific extensions to `core` for Windows platforms.

#![unstable(feature = "core_os_windows", issue = "none")]
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
/// #![feature(codeview_annotation, core_os, core_os_windows)]
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

/// Use `__fastfail` to abort the process
///
/// In Windows 8 and later, this will terminate the process immediately without
/// running any in-process exception handlers. In earlier versions of Windows,
/// this sequence of instructions will be treated as an access violation, which
/// will still terminate the process but might run some exception handlers.
///
/// <https://docs.microsoft.com/en-us/cpp/intrinsics/fastfail>
#[inline(always)]
#[unstable(feature = "windows_fastfail", issue = "none")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub fn __fastfail(code: crate::ffi::c_uint) -> ! {
    // SAFETY: These assembly sequences are documented at
    // <https://docs.microsoft.com/en-us/cpp/intrinsics/fastfail>
    unsafe {
        cfg_select! {
            // Miri does not support inline assembly
            miri => crate::intrinsics::abort_immediate(),
            any(target_arch = "x86", target_arch = "x86_64") => {
                crate::arch::asm!("int 0x29", in("ecx") code, options(noreturn, nostack))
            }
            all(target_arch = "arm", target_feature = "thumb-mode") => {
                crate::arch::asm!(".inst 0xDEFB", in("r0") code, options(noreturn, nostack))
            }
            any(target_arch = "aarch64", target_arch = "arm64ec") => {
                crate::arch::asm!("brk 0xF003", in("x0") code, options(noreturn, nostack))
            }
        }
    }
}
