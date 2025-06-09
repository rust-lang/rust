//! C's "variable arguments"
//!
//! Better known as "varargs".

use crate::ffi::c_void;
#[allow(unused_imports)]
use crate::fmt;
use crate::marker::{PhantomData, PhantomInvariantLifetime};
use crate::ops::{Deref, DerefMut};

// The name is WIP, using `VaListImpl` for now.
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
        #[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
        #[derive(Debug)]
        #[lang = "va_list"]
        pub struct VaListImpl<'f> {
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
        pub struct VaListImpl<'f> {
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
        pub struct VaListImpl<'f> {
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
        pub struct VaListImpl<'f> {
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
        pub struct VaListImpl<'f> {
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
    // - any other target for which we don't specify the `VaListImpl` above
    //
    // In this implementation the `va_list` type is just an alias for an opaque pointer.
    // That pointer is probably just the next variadic argument on the caller's stack.
    _ => {
        /// Basic implementation of a `va_list`.
        #[repr(transparent)]
        #[lang = "va_list"]
        pub struct VaListImpl<'f> {
            ptr: *mut c_void,

            // Invariant over `'f`, so each `VaListImpl<'f>` object is tied to
            // the region of the function it's defined in
            _marker: PhantomInvariantLifetime<'f>,
        }

        impl<'f> fmt::Debug for VaListImpl<'f> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        pub struct VaList<'a, 'f: 'a> {
            inner: &'a mut VaListImpl<'f>,
            _marker: PhantomData<&'a mut VaListImpl<'f>>,
        }


        impl<'f> VaListImpl<'f> {
            /// Converts a [`VaListImpl`] into a [`VaList`] that is binary-compatible with C's `va_list`.
            #[inline]
            pub fn as_va_list<'a>(&'a mut self) -> VaList<'a, 'f> {
                VaList { inner: self, _marker: PhantomData }
            }
        }
    }

    _ => {
        /// A wrapper for a `va_list`
        #[repr(transparent)]
        #[derive(Debug)]
        pub struct VaList<'a, 'f: 'a> {
            inner: VaListImpl<'f>,
            _marker: PhantomData<&'a mut VaListImpl<'f>>,
        }

        impl<'f> VaListImpl<'f> {
            /// Converts a [`VaListImpl`] into a [`VaList`] that is binary-compatible with C's `va_list`.
            #[inline]
            pub fn as_va_list<'a>(&'a mut self) -> VaList<'a, 'f> {
                VaList { inner: VaListImpl { ..*self }, _marker: PhantomData }
            }
        }
    }
}

impl<'a, 'f: 'a> Deref for VaList<'a, 'f> {
    type Target = VaListImpl<'f>;

    #[inline]
    fn deref(&self) -> &VaListImpl<'f> {
        &self.inner
    }
}

impl<'a, 'f: 'a> DerefMut for VaList<'a, 'f> {
    #[inline]
    fn deref_mut(&mut self) -> &mut VaListImpl<'f> {
        &mut self.inner
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

/// Trait which permits the allowed types to be used with [`VaListImpl::arg`].
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

impl<'f> VaListImpl<'f> {
    /// Advance to the next arg.
    #[inline]
    pub unsafe fn arg<T: VaArgSafe>(&mut self) -> T {
        // SAFETY: the caller must uphold the safety contract for `va_arg`.
        unsafe { va_arg(self) }
    }

    /// Copies the `va_list` at the current location.
    pub unsafe fn with_copy<F, R>(&self, f: F) -> R
    where
        F: for<'copy> FnOnce(VaList<'copy, 'f>) -> R,
    {
        let mut ap = self.clone();
        let ret = f(ap.as_va_list());
        // SAFETY: the caller must uphold the safety contract for `va_end`.
        unsafe {
            va_end(&mut ap);
        }
        ret
    }
}

impl<'f> Clone for VaListImpl<'f> {
    #[inline]
    fn clone(&self) -> Self {
        let mut dest = crate::mem::MaybeUninit::uninit();
        // SAFETY: we write to the `MaybeUninit`, thus it is initialized and `assume_init` is legal
        unsafe {
            va_copy(dest.as_mut_ptr(), self);
            dest.assume_init()
        }
    }
}

impl<'f> Drop for VaListImpl<'f> {
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
unsafe fn va_end(ap: &mut VaListImpl<'_>);

/// Copies the current location of arglist `src` to the arglist `dst`.
#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn va_copy<'f>(dest: *mut VaListImpl<'f>, src: &VaListImpl<'f>);

/// Loads an argument of type `T` from the `va_list` `ap` and increment the
/// argument `ap` points to.
#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn va_arg<T: VaArgSafe>(ap: &mut VaListImpl<'_>) -> T;
