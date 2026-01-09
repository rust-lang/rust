//! C's "variable arguments"
//!
//! Better known as "varargs".

#[cfg(not(target_arch = "xtensa"))]
use crate::ffi::c_void;
use crate::fmt;
use crate::intrinsics::{va_arg, va_copy, va_end};
use crate::marker::PhantomCovariantLifetime;

// There are currently three flavors of how a C `va_list` is implemented for
// targets that Rust supports:
//
// - `va_list` is an opaque pointer
// - `va_list` is a struct
// - `va_list` is a single-element array, containing a struct
//
// The opaque pointer approach is the simplest to implement: the pointer just
// points to an array of arguments on the caller's stack.
//
// The struct and single-element array variants are more complex, but
// potentially more efficient because the additional state makes it
// possible to pass variadic arguments via registers.
//
// The Rust `VaList` type is ABI-compatible with the C `va_list`.
// The struct and pointer cases straightforwardly map to their Rust equivalents,
// but the single-element array case is special: in C, this type is subject to
// array-to-pointer decay.
//
// The `#[rustc_pass_indirectly_in_non_rustic_abis]` attribute is used to match
// the pointer decay behavior in Rust, while otherwise matching Rust semantics.
// This attribute ensures that the compiler uses the correct ABI for functions
// like `extern "C" fn takes_va_list(va: VaList<'_>)` by passing `va` indirectly.
//
// The Clang `BuiltinVaListKind` enumerates the `va_list` variations that Clang supports,
// and we mirror these here.
//
// For all current LLVM targets, `va_copy` lowers to `memcpy`. Hence the inner structs below all
// derive `Copy`. However, in the future we might want to support a target where `va_copy`
// allocates, or otherwise violates the requirements of `Copy`. Therefore `VaList` is only `Clone`.
crate::cfg_select! {
    all(
        target_arch = "aarch64",
        not(target_vendor = "apple"),
        not(target_os = "uefi"),
        not(windows),
    ) => {
        /// AArch64 ABI implementation of a `va_list`.
        ///
        /// See the [AArch64 Procedure Call Standard] for more details.
        ///
        /// `va_copy` is `memcpy`: <https://github.com/llvm/llvm-project/blob/5aee01a3df011e660f26660bc30a8c94a1651d8e/llvm/lib/Target/AArch64/AArch64ISelLowering.cpp#L12682-L12700>
        ///
        /// [AArch64 Procedure Call Standard]:
        /// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
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
        ///
        /// See the [LLVM source] and [GCC header] for more details.
        ///
        /// `va_copy` is `memcpy`: <https://github.com/llvm/llvm-project/blob/5aee01a3df011e660f26660bc30a8c94a1651d8e/llvm/lib/Target/PowerPC/PPCISelLowering.cpp#L3755-L3764>
        ///
        /// [LLVM source]:
        /// https://github.com/llvm/llvm-project/blob/af9a4263a1a209953a1d339ef781a954e31268ff/llvm/lib/Target/PowerPC/PPCISelLowering.cpp#L4089-L4111
        /// [GCC header]: https://web.mit.edu/darwin/src/modules/gcc/gcc/ginclude/va-ppc.h
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
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
        ///
        /// See the [S/390x ELF Application Binary Interface Supplement] for more details.
        ///
        /// `va_copy` is `memcpy`: <https://github.com/llvm/llvm-project/blob/5aee01a3df011e660f26660bc30a8c94a1651d8e/llvm/lib/Target/SystemZ/SystemZISelLowering.cpp#L4457-L4472>
        ///
        /// [S/390x ELF Application Binary Interface Supplement]:
        /// https://docs.google.com/gview?embedded=true&url=https://github.com/IBM/s390x-abi/releases/download/v1.7/lzsabi_s390x.pdf
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        struct VaListInner {
            gpr: i64,
            fpr: i64,
            overflow_arg_area: *const c_void,
            reg_save_area: *const c_void,
        }
    }
    all(target_arch = "x86_64", not(target_os = "uefi"), not(windows)) => {
        /// x86_64 System V ABI implementation of a `va_list`.
        ///
        /// See the [System V AMD64 ABI] for more details.
        ///
        /// `va_copy` is `memcpy`: <https://github.com/llvm/llvm-project/blob/5aee01a3df011e660f26660bc30a8c94a1651d8e/llvm/lib/Target/X86/X86ISelLowering.cpp#26319>
        /// (github won't render that file, look for `SDValue LowerVACOPY`)
        ///
        /// [System V AMD64 ABI]:
        /// https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
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
        ///
        /// See the [LLVM source] for more details.
        ///
        /// `va_copy` is `memcpy`: <https://github.com/llvm/llvm-project/blob/5aee01a3df011e660f26660bc30a8c94a1651d8e/llvm/lib/Target/Xtensa/XtensaISelLowering.cpp#L1260>
        ///
        /// [LLVM source]:
        /// https://github.com/llvm/llvm-project/blob/af9a4263a1a209953a1d339ef781a954e31268ff/llvm/lib/Target/Xtensa/XtensaISelLowering.cpp#L1211-L1215
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        struct VaListInner {
            stk: *const i32,
            reg: *const i32,
            ndx: i32,
        }
    }

    all(target_arch = "hexagon", target_env = "musl") => {
        /// Hexagon Musl implementation of a `va_list`.
        ///
        /// See the [LLVM source] for more details. On bare metal Hexagon uses an opaque pointer.
        ///
        /// `va_copy` is `memcpy`: <https://github.com/llvm/llvm-project/blob/5aee01a3df011e660f26660bc30a8c94a1651d8e/llvm/lib/Target/Hexagon/HexagonISelLowering.cpp#L1087-L1102>
        ///
        /// [LLVM source]:
        /// https://github.com/llvm/llvm-project/blob/0cdc1b6dd4a870fc41d4b15ad97e0001882aba58/clang/lib/CodeGen/Targets/Hexagon.cpp#L407-L417
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        #[rustc_pass_indirectly_in_non_rustic_abis]
        struct VaListInner {
            __current_saved_reg_area_pointer: *const c_void,
            __saved_reg_area_end_pointer: *const c_void,
            __overflow_area_pointer: *const c_void,
        }
    }

    // The fallback implementation, used for:
    //
    // - apple aarch64 (see https://github.com/rust-lang/rust/pull/56599)
    // - windows
    // - powerpc64 & powerpc64le
    // - uefi
    // - any other target for which we don't specify the `VaListInner` above
    //
    // In this implementation the `va_list` type is just an alias for an opaque pointer.
    // That pointer is probably just the next variadic argument on the caller's stack.
    _ => {
        /// Basic implementation of a `va_list`.
        ///
        /// `va_copy` is `memcpy`: <https://github.com/llvm/llvm-project/blob/87e8e7d8f0db53060ef2f6ef4ab612fc0f2b4490/llvm/lib/Transforms/IPO/ExpandVariadics.cpp#L127-L129>
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy)]
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

impl VaList<'_> {
    // Helper used in the implementation of the `va_copy` intrinsic.
    pub(crate) fn duplicate(&self) -> Self {
        Self { inner: self.inner.clone(), _marker: self._marker }
    }
}

impl Clone for VaList<'_> {
    #[inline]
    fn clone(&self) -> Self {
        // We only implement Clone and not Copy because some future target might not be able to
        // implement Copy (e.g. because it allocates). For the same reason we use an intrinsic
        // to do the copying: the fact that on all current targets, this is just `memcpy`, is an implementation
        // detail. The intrinsic lets Miri catch UB from code incorrectly relying on that implementation detail.
        va_copy(self)
    }
}

impl<'f> Drop for VaList<'f> {
    fn drop(&mut self) {
        // SAFETY: this variable argument list is being dropped, so won't be read from again.
        unsafe { va_end(self) }
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
    /// This function is only sound to call when:
    ///
    /// - there is a next variable argument available.
    /// - the next argument's type must be ABI-compatible with the type `T`.
    /// - the next argument must have a properly initialized value of type `T`.
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

// Checks (via an assert in `compiler/rustc_ty_utils/src/abi.rs`) that the C ABI for the current
// target correctly implements `rustc_pass_indirectly_in_non_rustic_abis`.
const _: () = {
    #[repr(C)]
    #[rustc_pass_indirectly_in_non_rustic_abis]
    struct Type(usize);

    const extern "C" fn c(_: Type) {}

    c(Type(0))
};
