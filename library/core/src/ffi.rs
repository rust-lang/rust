#![stable(feature = "", since = "1.30.0")]
#![allow(non_camel_case_types)]

//! Utilities related to foreign function interface (FFI) bindings.

use crate::fmt;
use crate::marker::PhantomData;
use crate::ops::{Deref, DerefMut};

/// Equivalent to C's `void` type when used as a [pointer].
///
/// In essence, `*const c_void` is equivalent to C's `const void*`
/// and `*mut c_void` is equivalent to C's `void*`. That said, this is
/// *not* the same as C's `void` return type, which is Rust's `()` type.
///
/// To model pointers to opaque types in FFI, until `extern type` is
/// stabilized, it is recommended to use a newtype wrapper around an empty
/// byte array. See the [Nomicon] for details.
///
/// One could use `std::os::raw::c_void` if they want to support old Rust
/// compiler down to 1.1.0. After Rust 1.30.0, it was re-exported by
/// this definition. For more information, please read [RFC 2521].
///
/// [pointer]: ../../std/primitive.pointer.html
/// [Nomicon]: https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
/// [RFC 2521]: https://github.com/rust-lang/rfcs/blob/master/text/2521-c_void-reunification.md
// N.B., for LLVM to recognize the void pointer type and by extension
//     functions like malloc(), we need to have it represented as i8* in
//     LLVM bitcode. The enum used here ensures this and prevents misuse
//     of the "raw" type by only having private variants. We need two
//     variants, because the compiler complains about the repr attribute
//     otherwise and we need at least one variant as otherwise the enum
//     would be uninhabited and at least dereferencing such pointers would
//     be UB.
#[repr(u8)]
#[stable(feature = "core_c_void", since = "1.30.0")]
pub enum c_void {
    #[unstable(
        feature = "c_void_variant",
        reason = "temporary implementation detail",
        issue = "none"
    )]
    #[doc(hidden)]
    __variant1,
    #[unstable(
        feature = "c_void_variant",
        reason = "temporary implementation detail",
        issue = "none"
    )]
    #[doc(hidden)]
    __variant2,
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for c_void {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("c_void")
    }
}

/// Basic implementation of a `va_list`.
// The name is WIP, using `VaListImpl` for now.
#[cfg(any(
    all(not(target_arch = "aarch64"), not(target_arch = "powerpc"), not(target_arch = "x86_64")),
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")),
    target_arch = "wasm32",
    target_arch = "asmjs",
    windows
))]
#[repr(transparent)]
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
#[lang = "va_list"]
pub struct VaListImpl<'f> {
    ptr: *mut c_void,

    // Invariant over `'f`, so each `VaListImpl<'f>` object is tied to
    // the region of the function it's defined in
    _marker: PhantomData<&'f mut &'f c_void>,
}

#[cfg(any(
    all(not(target_arch = "aarch64"), not(target_arch = "powerpc"), not(target_arch = "x86_64")),
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")),
    target_arch = "wasm32",
    target_arch = "asmjs",
    windows
))]
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
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
    not(any(target_os = "macos", target_os = "ios")),
    not(windows)
))]
#[repr(C)]
#[derive(Debug)]
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
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
#[cfg(all(target_arch = "powerpc", not(windows)))]
#[repr(C)]
#[derive(Debug)]
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
#[lang = "va_list"]
pub struct VaListImpl<'f> {
    gpr: u8,
    fpr: u8,
    reserved: u16,
    overflow_arg_area: *mut c_void,
    reg_save_area: *mut c_void,
    _marker: PhantomData<&'f mut &'f c_void>,
}

/// x86_64 ABI implementation of a `va_list`.
#[cfg(all(target_arch = "x86_64", not(windows)))]
#[repr(C)]
#[derive(Debug)]
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
#[lang = "va_list"]
pub struct VaListImpl<'f> {
    gp_offset: i32,
    fp_offset: i32,
    overflow_arg_area: *mut c_void,
    reg_save_area: *mut c_void,
    _marker: PhantomData<&'f mut &'f c_void>,
}

/// A wrapper for a `va_list`
#[repr(transparent)]
#[derive(Debug)]
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
pub struct VaList<'a, 'f: 'a> {
    #[cfg(any(
        all(
            not(target_arch = "aarch64"),
            not(target_arch = "powerpc"),
            not(target_arch = "x86_64")
        ),
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")),
        target_arch = "wasm32",
        target_arch = "asmjs",
        windows
    ))]
    inner: VaListImpl<'f>,

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "powerpc", target_arch = "x86_64"),
        any(not(target_arch = "aarch64"), not(any(target_os = "macos", target_os = "ios"))),
        not(target_arch = "wasm32"),
        not(target_arch = "asmjs"),
        not(windows)
    ))]
    inner: &'a mut VaListImpl<'f>,

    _marker: PhantomData<&'a mut VaListImpl<'f>>,
}

#[cfg(any(
    all(not(target_arch = "aarch64"), not(target_arch = "powerpc"), not(target_arch = "x86_64")),
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")),
    target_arch = "wasm32",
    target_arch = "asmjs",
    windows
))]
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
impl<'f> VaListImpl<'f> {
    /// Convert a `VaListImpl` into a `VaList` that is binary-compatible with C's `va_list`.
    #[inline]
    pub fn as_va_list<'a>(&'a mut self) -> VaList<'a, 'f> {
        VaList { inner: VaListImpl { ..*self }, _marker: PhantomData }
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "powerpc", target_arch = "x86_64"),
    any(not(target_arch = "aarch64"), not(any(target_os = "macos", target_os = "ios"))),
    not(target_arch = "wasm32"),
    not(target_arch = "asmjs"),
    not(windows)
))]
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
impl<'f> VaListImpl<'f> {
    /// Convert a `VaListImpl` into a `VaList` that is binary-compatible with C's `va_list`.
    #[inline]
    pub fn as_va_list<'a>(&'a mut self) -> VaList<'a, 'f> {
        VaList { inner: self, _marker: PhantomData }
    }
}

#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
impl<'a, 'f: 'a> Deref for VaList<'a, 'f> {
    type Target = VaListImpl<'f>;

    #[inline]
    fn deref(&self) -> &VaListImpl<'f> {
        &self.inner
    }
}

#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
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
    #[unstable(
        feature = "c_variadic",
        reason = "the `c_variadic` feature has not been properly tested on \
                  all supported platforms",
        issue = "44930"
    )]
    pub trait VaArgSafe {}
}

macro_rules! impl_va_arg_safe {
    ($($t:ty),+) => {
        $(
            #[unstable(feature = "c_variadic",
                       reason = "the `c_variadic` feature has not been properly tested on \
                                 all supported platforms",
                       issue = "44930")]
            impl sealed_trait::VaArgSafe for $t {}
        )+
    }
}

impl_va_arg_safe! {i8, i16, i32, i64, usize}
impl_va_arg_safe! {u8, u16, u32, u64, isize}
impl_va_arg_safe! {f64}

#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
impl<T> sealed_trait::VaArgSafe for *mut T {}
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
impl<T> sealed_trait::VaArgSafe for *const T {}

#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
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

#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
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

#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
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

extern "rust-intrinsic" {
    /// Destroy the arglist `ap` after initialization with `va_start` or
    /// `va_copy`.
    fn va_end(ap: &mut VaListImpl<'_>);

    /// Copies the current location of arglist `src` to the arglist `dst`.
    fn va_copy<'f>(dest: *mut VaListImpl<'f>, src: &VaListImpl<'f>);

    /// Loads an argument of type `T` from the `va_list` `ap` and increment the
    /// argument `ap` points to.
    fn va_arg<T: sealed_trait::VaArgSafe>(ap: &mut VaListImpl<'_>) -> T;
}
