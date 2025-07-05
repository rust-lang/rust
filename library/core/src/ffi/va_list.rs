//! C's "variable arguments"
//!
//! Better known as "varargs".

use crate::ffi::c_void;
#[allow(unused_imports)]
use crate::fmt;
use crate::marker::PhantomInvariantLifetime;

// Most targets explicitly specify the layout of `va_list`, this layout is matched here.
// For `va_list`s which are single-element array in C (and therefore experience array-to-pointer
// decay when passed as arguments in C), the `VaList` struct is annotated with
// `#[rustc_pass_indirectly_in_non_rustic_abis]`. This ensures that the compiler uses the correct
// ABI for functions like `extern "C" fn takes_va_list(va: VaList<'_>)` by passing `va` indirectly.

// Note that currently support for `#[rustc_pass_indirectly_in_non_rustic_abis]` is only implemented
// on architectures which need it here, so when adding support for a new architecture the following
// will need to happen:
//
// - Check that the calling conventions used on the new architecture correctly check
//   `arg.layout.pass_indirectly_in_non_rustic_abis()` and call `arg.make_indirect()` if it returns
//   `true`.
// - Add a revision to the `tests/ui/abi/pass-indirectly-attr.rs` test for the new architecture.
// - Add the new architecture to the `supported_architectures` array in the
//   `check_pass_indirectly_in_non_rustic_abis` function in
//   `compiler/rustc_passes/src/check_attr.rs`. This will stop the compiler from emitting an error
//   message when the attribute is used on that architecture.
crate::cfg_select! {
    all(
        target_arch = "aarch64",
        not(target_vendor = "apple"),
        not(target_os = "uefi"),
        not(windows),
    ) => {
        /// AArch64 ABI implementation of a `va_list`. See the
        /// [AArch64 Procedure Call Standard] for more details.
        ///
        /// [AArch64 Procedure Call Standard]:
        /// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf
        #[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
        #[derive(Debug)]
        #[lang = "va_list"]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        pub struct VaList<'f> {
            stack: *mut c_void,
            gr_top: *mut c_void,
            vr_top: *mut c_void,
            gr_offs: i32,
            vr_offs: i32,
            _marker: PhantomInvariantLifetime<'f>,
        }
    }
    all(target_arch = "powerpc", not(target_os = "uefi"), not(windows)) => {
        /// PowerPC ABI implementation of a `va_list`.
        #[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
        #[derive(Debug)]
        #[lang = "va_list"]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        pub struct VaList<'f> {
            gpr: u8,
            fpr: u8,
            reserved: u16,
            overflow_arg_area: *mut c_void,
            reg_save_area: *mut c_void,
            _marker: PhantomInvariantLifetime<'f>,
        }
    }
    target_arch = "s390x" => {
        /// s390x ABI implementation of a `va_list`.
        #[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
        #[derive(Debug)]
        #[lang = "va_list"]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        pub struct VaList<'f> {
            gpr: i64,
            fpr: i64,
            overflow_arg_area: *mut c_void,
            reg_save_area: *mut c_void,
            _marker: PhantomInvariantLifetime<'f>,
        }
    }
    all(target_arch = "x86_64", not(target_os = "uefi"), not(windows)) => {
        /// x86_64 ABI implementation of a `va_list`.
        #[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
        #[derive(Debug)]
        #[lang = "va_list"]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        pub struct VaList<'f> {
            gp_offset: i32,
            fp_offset: i32,
            overflow_arg_area: *mut c_void,
            reg_save_area: *mut c_void,
            _marker: PhantomInvariantLifetime<'f>,
        }
    }
    target_arch = "xtensa" => {
        /// Xtensa ABI implementation of a `va_list`.
        #[repr(C)]
        #[derive(Debug)]
        #[lang = "va_list"]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        pub struct VaList<'f> {
            stk: *mut i32,
            reg: *mut i32,
            ndx: i32,
            _marker: PhantomInvariantLifetime<'f>,
        }
    }

    // The fallback implementation, used for:
    //
    // - apple aarch64 (see https://github.com/rust-lang/rust/pull/56599)
    // - windows
    // - uefi
    // - any other target for which we don't specify the `VaList` above
    //
    // In this implementation the `va_list` type is just an alias for an opaque pointer.
    // That pointer is probably just the next variadic argument on the caller's stack.
    _ => {
        /// Basic implementation of a `va_list`.
        #[repr(transparent)]
        #[lang = "va_list"]
        pub struct VaList<'f> {
            ptr: *mut c_void,

            // Invariant over `'f`, so each `VaList<'f>` object is tied to
            // the region of the function it's defined in
            _marker: PhantomInvariantLifetime<'f>,
        }

        impl<'f> fmt::Debug for VaList<'f> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple("VaList").field(&self.ptr).finish()
            }
        }
    }
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for isize {}

    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for usize {}

    impl Sealed for f64 {}

    impl<T> Sealed for *mut T {}
    impl<T> Sealed for *const T {}
}

/// Trait which permits the allowed types to be used with [`VaList::arg`].
///
/// # Safety
///
/// This trait must only be implemented for types that C passes as varargs without implicit promotion.
///
/// In C varargs, integers smaller than [`c_int`] and floats smaller than [`c_double`]
/// are implicitly promoted to [`c_int`] and [`c_double`] respectively. Implementing this trait for
/// types that are subject to this promotion rule is invalid.
///
/// [`c_int`]: core::ffi::c_int
/// [`c_double`]: core::ffi::c_double
pub unsafe trait VaArgSafe: sealed::Sealed {}

// i8 and i16 are implicitly promoted to c_int in C, and cannot implement `VaArgSafe`.
unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
unsafe impl VaArgSafe for isize {}

// u8 and u16 are implicitly promoted to c_int in C, and cannot implement `VaArgSafe`.
unsafe impl VaArgSafe for u32 {}
unsafe impl VaArgSafe for u64 {}
unsafe impl VaArgSafe for usize {}

// f32 is implicitly promoted to c_double in C, and cannot implement `VaArgSafe`.
unsafe impl VaArgSafe for f64 {}

unsafe impl<T> VaArgSafe for *mut T {}
unsafe impl<T> VaArgSafe for *const T {}

impl<'f> VaList<'f> {
    /// Advance to the next arg.
    #[inline]
    pub unsafe fn arg<T: VaArgSafe>(&mut self) -> T {
        // SAFETY: the caller must uphold the safety contract for `va_arg`.
        unsafe { crate::intrinsics::va_arg(self) }
    }
}

impl<'f> Clone for VaList<'f> {
    #[inline]
    fn clone(&self) -> Self {
        let mut dest = crate::mem::MaybeUninit::uninit();
        // SAFETY: we write to the `MaybeUninit`, thus it is initialized and `assume_init` is legal.
        unsafe {
            crate::intrinsics::va_copy(dest.as_mut_ptr(), self);
            dest.assume_init()
        }
    }
}

impl<'f> Drop for VaList<'f> {
    fn drop(&mut self) {
        // Rust requires that not calling `va_end` on a `va_list` does not cause undefined behaviour
        // (as it is safe to leak values). As `va_end` is a no-op on all current LLVM targets, this
        // destructor is empty.
    }
}
