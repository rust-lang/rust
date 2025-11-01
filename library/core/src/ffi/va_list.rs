//! C's "variable arguments"
//!
//! Better known as "varargs".

#[cfg(not(target_arch = "xtensa"))]
use crate::ffi::c_void;
use crate::fmt;
use crate::intrinsics::{va_arg, va_copy};
use crate::marker::PhantomCovariantLifetime;

// Most targets explicitly specify the layout of `va_list`, this layout is matched here.
// For `va_list`s which are single-element array in C (and therefore experience array-to-pointer
// decay when passed as arguments in C), the `VaList` struct is annotated with
// `#[rustc_pass_indirectly_in_non_rustic_abis]`. This ensures that the compiler uses the correct
// ABI for functions like `extern "C" fn takes_va_list(va: VaList<'_>)` by passing `va` indirectly.
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
        #[repr(C)]
        #[derive(Debug)]
        struct VaListInner {
            stack: *const c_void,
            gr_top: *const c_void,
            vr_top: *const c_void,
            gr_offs: i32,
            vr_offs: i32,
        }
    }
    all(target_arch = "powerpc", not(target_os = "uefi"), not(windows)) => {
        /// PowerPC ABI implementation of a `va_list`.
        #[repr(C)]
        #[derive(Debug)]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        struct VaListInner {
            gpr: u8,
            fpr: u8,
            reserved: u16,
            overflow_arg_area: *const c_void,
            reg_save_area: *const c_void,
        }
    }
    target_arch = "s390x" => {
        /// s390x ABI implementation of a `va_list`.
        #[repr(C)]
        #[derive(Debug)]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        struct VaListInner {
            gpr: i64,
            fpr: i64,
            overflow_arg_area: *const c_void,
            reg_save_area: *const c_void,
        }
    }
    all(target_arch = "x86_64", not(target_os = "uefi"), not(windows)) => {
        /// x86_64 ABI implementation of a `va_list`.
        #[repr(C)]
        #[derive(Debug)]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        struct VaListInner {
            gp_offset: i32,
            fp_offset: i32,
            overflow_arg_area: *const c_void,
            reg_save_area: *const c_void,
        }
    }
    target_arch = "xtensa" => {
        /// Xtensa ABI implementation of a `va_list`.
        #[repr(C)]
        #[derive(Debug)]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        struct VaListInner {
            stk: *const i32,
            reg: *const i32,
            ndx: i32,
        }
    }

    // The fallback implementation, used for:
    //
    // - apple aarch64 (see https://github.com/rust-lang/rust/pull/56599)
    // - windows
    // - uefi
    // - any other target for which we don't specify the `VaListInner` above
    //
    // In this implementation the `va_list` type is just an alias for an opaque pointer.
    // That pointer is probably just the next variadic argument on the caller's stack.
    _ => {
        /// Basic implementation of a `va_list`.
        #[repr(transparent)]
        #[derive(Debug)]
        struct VaListInner {
            ptr: *const c_void,
        }
    }
}

/// A variable argument list, equivalent to `va_list` in C.
#[repr(transparent)]
#[lang = "va_list"]
pub struct VaList<'a> {
    inner: VaListInner,
    _marker: PhantomCovariantLifetime<'a>,
}

impl fmt::Debug for VaList<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // No need to include `_marker` in debug output.
        f.debug_tuple("VaList").field(&self.inner).finish()
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

/// Types that are valid to read using [`VaList::arg`].
///
/// # Safety
///
/// The standard library implements this trait for primitive types that are
/// expected to have a variable argument application-binary interface (ABI) on all
/// platforms.
///
/// When C passes variable arguments, integers smaller than [`c_int`] and floats smaller
/// than [`c_double`] are implicitly promoted to [`c_int`] and [`c_double`] respectively.
/// Implementing this trait for types that are subject to this promotion rule is invalid.
///
/// [`c_int`]: core::ffi::c_int
/// [`c_double`]: core::ffi::c_double
// We may unseal this trait in the future, but currently our `va_arg` implementations don't support
// types with an alignment larger than 8, or with a non-scalar layout. Inline assembly can be used
// to accept unsupported types in the meantime.
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
    /// Advance to and read the next variable argument.
    ///
    /// # Safety
    ///
    /// This function is only sound to call when the next variable argument:
    ///
    /// - has a type that is ABI-compatible with the type `T`
    /// - has a value that is a properly initialized value of type `T`
    ///
    /// Calling this function with an incompatible type, an invalid value, or when there
    /// are no more variable arguments, is unsound.
    ///
    /// [valid]: https://doc.rust-lang.org/nightly/nomicon/what-unsafe-does.html
    #[inline]
    pub unsafe fn arg<T: VaArgSafe>(&mut self) -> T {
        // SAFETY: the caller must uphold the safety contract for `va_arg`.
        unsafe { va_arg(self) }
    }
}

impl<'f> Clone for VaList<'f> {
    #[inline]
    fn clone(&self) -> Self {
        let mut dest = crate::mem::MaybeUninit::uninit();
        // SAFETY: we write to the `MaybeUninit`, thus it is initialized and `assume_init` is legal.
        unsafe {
            va_copy(dest.as_mut_ptr(), self);
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

// Checks (via an assert in `compiler/rustc_ty_utils/src/abi.rs`) that the C ABI for the current
// target correctly implements `rustc_pass_indirectly_in_non_rustic_abis`.
const _: () = {
    #[repr(C)]
    #[rustc_pass_indirectly_in_non_rustic_abis]
    struct Type(usize);

    const extern "C" fn c(_: Type) {}

    c(Type(0))
};
