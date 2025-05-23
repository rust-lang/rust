//! C's "variable arguments"
//!
//! Better known as "varargs".

use crate::ffi::c_void;
use crate::marker::PhantomInvariantLifetime;
use crate::mem::MaybeUninit;

// The name is WIP, using `VaListTag` for now.
//
// Most targets explicitly specify the layout of `va_list`, this layout is matched here.
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
        #[doc(hidden)]
        #[repr(C)]
        #[derive(Debug)]
        #[lang = "va_list_tag"]
        #[unstable(feature = "va_list_impl", issue = "none")]
        pub struct VaListTag<'a> {
            stack: *mut c_void,
            gr_top: *mut c_void,
            vr_top: *mut c_void,
            gr_offs: i32,
            vr_offs: i32,
            _marker: PhantomInvariantLifetime<'a>,
        }
    }
    all(target_arch = "powerpc", not(target_os = "uefi"), not(windows)) => {
        /// PowerPC ABI implementation of a `va_list`.
        #[doc(hidden)]
        #[repr(C)]
        #[derive(Debug)]
        #[lang = "va_list_tag"]
        #[unstable(feature = "va_list_impl", issue = "none")]
        pub struct VaListTag<'a> {
            gpr: u8,
            fpr: u8,
            reserved: u16,
            overflow_arg_area: *mut c_void,
            reg_save_area: *mut c_void,
            _marker: PhantomInvariantLifetime<'a>,
        }
    }
    target_arch = "s390x" => {
        /// s390x ABI implementation of a `va_list`.
        #[doc(hidden)]
        #[repr(C)]
        #[derive(Debug)]
        #[lang = "va_list_tag"]
        #[unstable(feature = "va_list_impl", issue = "none")]
        pub struct VaListTag<'a> {
            gpr: i64,
            fpr: i64,
            overflow_arg_area: *mut c_void,
            reg_save_area: *mut c_void,
            _marker: PhantomInvariantLifetime<'a>,
        }
    }
    all(target_arch = "x86_64", not(target_os = "uefi"), not(windows)) => {
        /// x86_64 ABI implementation of a `va_list`.
        #[doc(hidden)]
        #[repr(C)]
        #[derive(Debug)]
        #[lang = "va_list_tag"]
        #[unstable(feature = "va_list_impl", issue = "none")]
        pub struct VaListTag<'a> {
            gp_offset: i32,
            fp_offset: i32,
            overflow_arg_area: *mut c_void,
            reg_save_area: *mut c_void,
            _marker: PhantomInvariantLifetime<'a>,
        }
    }
    target_arch = "xtensa" => {
        /// Xtensa ABI implementation of a `va_list`.
        #[doc(hidden)]
        #[repr(C)]
        #[derive(Debug)]
        #[lang = "va_list_tag"]
        #[unstable(feature = "va_list_impl", issue = "none")]
        pub struct VaListTag<'a> {
            stk: *mut i32,
            reg: *mut i32,
            ndx: i32,
            _marker: PhantomInvariantLifetime<'a>,
        }
    }

    // The fallback implementation, used for:
    //
    // - apple aarch64 (see https://github.com/rust-lang/rust/pull/56599)
    // - windows
    // - uefi
    // - any other target for which we don't specify the `VaListTag` above
    //
    // In this implementation the `va_list` type is just an alias for an opaque pointer.
    // That pointer is probably just the next variadic argument on the caller's stack.
    _ => {
        /// Basic implementation of a `va_list`.
        #[doc(hidden)]
        #[repr(transparent)]
        #[lang = "va_list_tag"]
        #[unstable(feature = "va_list_impl", issue = "none")]
        pub struct VaListTag<'a> {
            ptr: *mut c_void,
            _marker: PhantomInvariantLifetime<'a>,
        }

        impl<'a> crate::fmt::Debug for VaListTag<'a> {
            fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
                write!(f, "va_list* {:p}", self.ptr)
            }
        }
    }
}

crate::cfg_select! {
    all(
        any(
            target_arch = "aarch64",
            target_arch = "powerpc",
            target_arch = "s390x",
            target_arch = "x86_64"
        ),
        not(target_arch = "xtensa"),
        any(not(target_arch = "aarch64"), not(target_vendor = "apple")),
        not(target_family = "wasm"),
        not(target_os = "uefi"),
        not(windows),
    ) => {
        /// A wrapper for a `va_list`
        #[repr(transparent)]
        #[derive(Debug)]
        #[lang = "va_list"]
        pub struct VaList<'a> {
            inner: &'a mut VaListTag<'a>,
        }

        impl<'a> VaList<'a> {
            #[doc(hidden)]
            #[unstable(feature = "va_list_impl", issue = "none")]
            #[inline(always)]
            pub fn copy<'b>(&self, tag: &'b mut MaybeUninit<VaListTag<'b>>) -> VaList<'b>
            where
                'a: 'b,
            {
                // The signature of `copy` already enforces the correct lifetime constraints.
                // `va_copy` is an intrinsic, which makes it less flexible lifetime-wise.
                let ptr = tag.as_mut_ptr().cast::<VaListTag<'a>>();

                // SAFETY: `tag` has sufficient space to store a `VaListTag`.
                unsafe { va_copy_intrinsic::va_copy(ptr, &self.inner) };

                // SAFETY: `va_copy` has initialized the tag.
                VaList { inner: unsafe { tag.assume_init_mut() } }
            }
        }

    }

    _ => {
        /// A wrapper for a `va_list`
        #[repr(transparent)]
        #[derive(Debug)]
        #[lang = "va_list"]
        pub struct VaList<'a> {
            inner: VaListTag<'a>,
        }


        impl<'a> VaList<'a> {
            #[doc(hidden)]
            #[unstable(feature = "va_list_impl", issue = "none")]
            #[inline(always)]
            pub fn copy<'b>(&self, tag: &'b mut MaybeUninit<VaListTag<'b>>) -> VaList<'b>
            where
                'a: 'b,
            {
                // The signature of `copy` already enforces the correct lifetime constraints.
                // `va_copy` is an intrinsic, which makes it less flexible lifetime-wise.
                let ptr = tag.as_mut_ptr().cast::<VaListTag<'a>>();

                // SAFETY: `tag` has sufficient space to store a `VaListTag`.
                unsafe { va_copy_intrinsic::va_copy(ptr, &self.inner) };

                // SAFETY: `va_copy` has initialized the tag.
                VaList { inner: unsafe { tag.assume_init_read() } }
            }
        }
    }
}

/// Copy a [`VaList`].
#[allow_internal_unstable(super_let)]
#[allow_internal_unstable(va_list_impl)]
pub macro va_copy($va_list:expr $(,)?) {{
    super let mut tag = $crate::mem::MaybeUninit::uninit();
    $va_list.copy(&mut tag)
}}

impl<'a> VaListTag<'a> {
    /// Advance to the next arg.
    #[inline(never)]
    unsafe fn arg<T: VaArgSafe>(&mut self) -> T {
        // SAFETY: the caller must uphold the safety contract for `va_arg`.
        unsafe { va_arg(self) }
    }
}

impl<'a> VaList<'a> {
    /// Advance to the next arg.
    #[inline]
    pub unsafe fn arg<T: VaArgSafe>(&mut self) -> T {
        // SAFETY: the caller must uphold the safety contract for `va_arg`.
        unsafe { self.inner.arg() }
    }

    /// Copies the `va_list` at the current location.
    pub unsafe fn with_copy<F, R>(&self, f: F) -> R
    where
        F: FnOnce(VaList<'_>) -> R,
    {
        let ap = va_copy!(self);
        let ret = f(ap);
        ret
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

impl Drop for VaListTag<'_> {
    fn drop(&mut self) {
        // FIXME: this should call `va_end`, but there's no clean way to
        // guarantee that `drop` always gets inlined into its caller,
        // so the `va_end` would get directly called from the same function as
        // the corresponding `va_copy`. `man va_end` states that C requires this,
        // and LLVM basically follows the C semantics, so we need to make sure
        // that `va_end` is always called from the same function as `va_copy`.
        // For more details, see https://github.com/rust-lang/rust/pull/59625
        // and https://llvm.org/docs/LangRef.html#llvm-va-end-intrinsic.
        //
        // This works for now, since `va_end` is a no-op on all current LLVM targets.
    }
}

/// Destroy the arglist `ap` after initialization with `va_start` or
/// `va_copy`.
#[rustc_intrinsic]
#[rustc_nounwind]
#[allow(unused)]
unsafe fn va_end(ap: &mut VaListTag<'_>);

// This intrinsic is in a module so that its name does not clash with the `va_copy!` macro.
mod va_copy_intrinsic {
    use super::VaListTag;
    /// Copies the current location of arglist `src` to the arglist `dst`.
    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub(super) unsafe fn va_copy<'a>(dest: *mut VaListTag<'a>, src: &VaListTag<'a>);
}

/// Loads an argument of type `T` from the `va_list` `ap` and increment the
/// argument `ap` points to.
#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn va_arg<T: VaArgSafe>(ap: &mut VaListTag<'_>) -> T;
