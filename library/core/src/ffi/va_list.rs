//! C's "variable arguments"
//!
//! Better known as "varargs".

use crate::ffi::c_void;
#[allow(unused_imports)]
use crate::fmt;
use crate::marker::PhantomData;
use crate::ops::{Deref, DerefMut};

/// Basic implementation of a `va_list`.
// The name is WIP, using `VaListImpl` for now.
#[cfg(any(
    all(
        not(target_arch = "aarch64"),
        not(target_arch = "powerpc"),
        not(target_arch = "s390x"),
        not(target_arch = "xtensa"),
        not(target_arch = "x86_64")
    ),
    all(target_arch = "aarch64", target_vendor = "apple"),
    target_family = "wasm",
    target_os = "uefi",
    windows,
))]
#[repr(transparent)]
#[lang = "va_list"]
pub struct VaListImpl<'f> {
    ptr: *mut c_void,

    // Invariant over `'f`, so each `VaListImpl<'f>` object is tied to
    // the region of the function it's defined in
    _marker: PhantomData<&'f mut &'f c_void>,
}

#[cfg(any(
    all(
        not(target_arch = "aarch64"),
        not(target_arch = "powerpc"),
        not(target_arch = "s390x"),
        not(target_arch = "xtensa"),
        not(target_arch = "x86_64")
    ),
    all(target_arch = "aarch64", target_vendor = "apple"),
    target_family = "wasm",
    target_os = "uefi",
    windows,
))]
impl<'f> fmt::Debug for VaListImpl<'f> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "va_list* {:p}", self.ptr)
    }
}

/// AArch64 ABI implementation of a `va_list`. See the
/// [AArch64 Procedure Call Standard] for more details.
///
/// [AArch64 Procedure Call Standard]:
/// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf
#[cfg(all(
    target_arch = "aarch64",
    not(target_vendor = "apple"),
    not(target_os = "uefi"),
    not(windows),
))]
#[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
#[derive(Debug)]
#[lang = "va_list"]
pub struct VaListImpl<'f> {
    stack: *mut c_void,
    gr_top: *mut c_void,
    vr_top: *mut c_void,
    gr_offs: i32,
    vr_offs: i32,
    _marker: PhantomData<&'f mut &'f c_void>,
}

/// PowerPC ABI implementation of a `va_list`.
#[cfg(all(target_arch = "powerpc", not(target_os = "uefi"), not(windows)))]
#[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
#[derive(Debug)]
#[lang = "va_list"]
pub struct VaListImpl<'f> {
    gpr: u8,
    fpr: u8,
    reserved: u16,
    overflow_arg_area: *mut c_void,
    reg_save_area: *mut c_void,
    _marker: PhantomData<&'f mut &'f c_void>,
}

/// s390x ABI implementation of a `va_list`.
#[cfg(target_arch = "s390x")]
#[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
#[derive(Debug)]
#[lang = "va_list"]
pub struct VaListImpl<'f> {
    gpr: i64,
    fpr: i64,
    overflow_arg_area: *mut c_void,
    reg_save_area: *mut c_void,
    _marker: PhantomData<&'f mut &'f c_void>,
}

/// x86_64 ABI implementation of a `va_list`.
#[cfg(all(target_arch = "x86_64", not(target_os = "uefi"), not(windows)))]
#[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
#[derive(Debug)]
#[lang = "va_list"]
pub struct VaListImpl<'f> {
    gp_offset: i32,
    fp_offset: i32,
    overflow_arg_area: *mut c_void,
    reg_save_area: *mut c_void,
    _marker: PhantomData<&'f mut &'f c_void>,
}

/// Xtensa ABI implementation of a `va_list`.
#[cfg(target_arch = "xtensa")]
#[repr(C)]
#[derive(Debug)]
#[lang = "va_list"]
pub struct VaListImpl<'f> {
    stk: *mut i32,
    reg: *mut i32,
    ndx: i32,
    _marker: PhantomData<&'f mut &'f c_void>,
}

/// A wrapper for a `va_list`
#[repr(transparent)]
#[derive(Debug)]
pub struct VaList<'a, 'f: 'a> {
    #[cfg(any(
        all(
            not(target_arch = "aarch64"),
            not(target_arch = "powerpc"),
            not(target_arch = "s390x"),
            not(target_arch = "x86_64")
        ),
        target_arch = "xtensa",
        all(target_arch = "aarch64", target_vendor = "apple"),
        target_family = "wasm",
        target_os = "uefi",
        windows,
    ))]
    inner: VaListImpl<'f>,

    #[cfg(all(
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
    ))]
    inner: &'a mut VaListImpl<'f>,

    _marker: PhantomData<&'a mut VaListImpl<'f>>,
}

#[cfg(any(
    all(
        not(target_arch = "aarch64"),
        not(target_arch = "powerpc"),
        not(target_arch = "s390x"),
        not(target_arch = "x86_64")
    ),
    target_arch = "xtensa",
    all(target_arch = "aarch64", target_vendor = "apple"),
    target_family = "wasm",
    target_os = "uefi",
    windows,
))]
impl<'f> VaListImpl<'f> {
    /// Converts a `VaListImpl` into a `VaList` that is binary-compatible with C's `va_list`.
    #[inline]
    pub fn as_va_list<'a>(&'a mut self) -> VaList<'a, 'f> {
        VaList { inner: VaListImpl { ..*self }, _marker: PhantomData }
    }
}

#[cfg(all(
    any(
        target_arch = "aarch64",
        target_arch = "powerpc",
        target_arch = "s390x",
        target_arch = "xtensa",
        target_arch = "x86_64"
    ),
    not(target_arch = "xtensa"),
    any(not(target_arch = "aarch64"), not(target_vendor = "apple")),
    not(target_family = "wasm"),
    not(target_os = "uefi"),
    not(windows),
))]
impl<'f> VaListImpl<'f> {
    /// Converts a `VaListImpl` into a `VaList` that is binary-compatible with C's `va_list`.
    #[inline]
    pub fn as_va_list<'a>(&'a mut self) -> VaList<'a, 'f> {
        VaList { inner: self, _marker: PhantomData }
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

// The VaArgSafe trait needs to be used in public interfaces, however, the trait
// itself must not be allowed to be used outside this module. Allowing users to
// implement the trait for a new type (thereby allowing the va_arg intrinsic to
// be used on a new type) is likely to cause undefined behavior.
//
// FIXME(dlrobertson): In order to use the VaArgSafe trait in a public interface
// but also ensure it cannot be used elsewhere, the trait needs to be public
// within a private module. Once RFC 2145 has been implemented look into
// improving this.
mod sealed_trait {
    /// Trait which permits the allowed types to be used with [super::VaListImpl::arg].
    pub unsafe trait VaArgSafe {}
}

macro_rules! impl_va_arg_safe {
    ($($t:ty),+) => {
        $(
            unsafe impl sealed_trait::VaArgSafe for $t {}
        )+
    }
}

impl_va_arg_safe! {i8, i16, i32, i64, usize}
impl_va_arg_safe! {u8, u16, u32, u64, isize}
impl_va_arg_safe! {f64}

unsafe impl<T> sealed_trait::VaArgSafe for *mut T {}
unsafe impl<T> sealed_trait::VaArgSafe for *const T {}

impl<'f> VaListImpl<'f> {
    /// Advance to the next arg.
    #[inline]
    pub unsafe fn arg<T: sealed_trait::VaArgSafe>(&mut self) -> T {
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
unsafe fn va_arg<T: sealed_trait::VaArgSafe>(ap: &mut VaListImpl<'_>) -> T;
