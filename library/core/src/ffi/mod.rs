//! Platform-specific types, as defined by C.
//!
//! Code that interacts via FFI will almost certainly be using the
//! base types provided by C, which aren't nearly as nicely defined
//! as Rust's primitive types. This module provides types which will
//! match those defined by C, so that code that interacts with C will
//! refer to the correct types.

#![stable(feature = "", since = "1.30.0")]
#![allow(non_camel_case_types)]

use crate::fmt;
use crate::marker::PhantomData;
use crate::num::*;
use crate::ops::{Deref, DerefMut};

#[stable(feature = "core_c_str", since = "1.64.0")]
pub use self::c_str::{CStr, FromBytesUntilNulError, FromBytesWithNulError};

mod c_str;

macro_rules! type_alias_no_nz {
    {
      $Docfile:tt, $Alias:ident = $Real:ty;
      $( $Cfg:tt )*
    } => {
        #[doc = include_str!($Docfile)]
        $( $Cfg )*
        #[stable(feature = "core_ffi_c", since = "1.64.0")]
        pub type $Alias = $Real;
    }
}

// To verify that the NonZero types in this file's macro invocations correspond
//
//  perl -n < library/std/src/os/raw/mod.rs -e 'next unless m/type_alias\!/; die "$_ ?" unless m/, (c_\w+) = (\w+), NonZero_(\w+) = NonZero(\w+)/; die "$_ ?" unless $3 eq $1 and $4 eq ucfirst $2'
//
// NB this does not check that the main c_* types are right.

macro_rules! type_alias {
    {
      $Docfile:tt, $Alias:ident = $Real:ty, $NZAlias:ident = $NZReal:ty;
      $( $Cfg:tt )*
    } => {
        type_alias_no_nz! { $Docfile, $Alias = $Real; $( $Cfg )* }

        #[doc = concat!("Type alias for `NonZero` version of [`", stringify!($Alias), "`]")]
        #[unstable(feature = "raw_os_nonzero", issue = "82363")]
        $( $Cfg )*
        pub type $NZAlias = $NZReal;
    }
}

type_alias! { "c_char.md", c_char = c_char_definition::c_char, NonZero_c_char = c_char_definition::NonZero_c_char;
// Make this type alias appear cfg-dependent so that Clippy does not suggest
// replacing `0 as c_char` with `0_i8`/`0_u8`. This #[cfg(all())] can be removed
// after the false positive in https://github.com/rust-lang/rust-clippy/issues/8093
// is fixed.
#[cfg(all())]
#[doc(cfg(all()))] }

type_alias! { "c_schar.md", c_schar = i8, NonZero_c_schar = NonZeroI8; }
type_alias! { "c_uchar.md", c_uchar = u8, NonZero_c_uchar = NonZeroU8; }
type_alias! { "c_short.md", c_short = i16, NonZero_c_short = NonZeroI16; }
type_alias! { "c_ushort.md", c_ushort = u16, NonZero_c_ushort = NonZeroU16; }

type_alias! { "c_int.md", c_int = c_int_definition::c_int, NonZero_c_int = c_int_definition::NonZero_c_int;
#[doc(cfg(all()))] }
type_alias! { "c_uint.md", c_uint = c_int_definition::c_uint, NonZero_c_uint = c_int_definition::NonZero_c_uint;
#[doc(cfg(all()))] }

type_alias! { "c_long.md", c_long = c_long_definition::c_long, NonZero_c_long = c_long_definition::NonZero_c_long;
#[doc(cfg(all()))] }
type_alias! { "c_ulong.md", c_ulong = c_long_definition::c_ulong, NonZero_c_ulong = c_long_definition::NonZero_c_ulong;
#[doc(cfg(all()))] }

type_alias! { "c_longlong.md", c_longlong = i64, NonZero_c_longlong = NonZeroI64; }
type_alias! { "c_ulonglong.md", c_ulonglong = u64, NonZero_c_ulonglong = NonZeroU64; }

type_alias_no_nz! { "c_float.md", c_float = f32; }
type_alias_no_nz! { "c_double.md", c_double = f64; }

/// Equivalent to C's `size_t` type, from `stddef.h` (or `cstddef` for C++).
///
/// This type is currently always [`usize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "88345")]
pub type c_size_t = usize;

/// Equivalent to C's `ptrdiff_t` type, from `stddef.h` (or `cstddef` for C++).
///
/// This type is currently always [`isize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "88345")]
pub type c_ptrdiff_t = isize;

/// Equivalent to C's `ssize_t` (on POSIX) or `SSIZE_T` (on Windows) type.
///
/// This type is currently always [`isize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "88345")]
pub type c_ssize_t = isize;

mod c_char_definition {
    cfg_if! {
        // These are the targets on which c_char is unsigned.
        if #[cfg(any(
            all(
                target_os = "linux",
                any(
                    target_arch = "aarch64",
                    target_arch = "arm",
                    target_arch = "hexagon",
                    target_arch = "powerpc",
                    target_arch = "powerpc64",
                    target_arch = "s390x",
                    target_arch = "riscv64",
                    target_arch = "riscv32"
                )
            ),
            all(target_os = "android", any(target_arch = "aarch64", target_arch = "arm")),
            all(target_os = "l4re", target_arch = "x86_64"),
            all(
                any(target_os = "freebsd", target_os = "openbsd"),
                any(
                    target_arch = "aarch64",
                    target_arch = "arm",
                    target_arch = "powerpc",
                    target_arch = "powerpc64",
                    target_arch = "riscv64"
                )
            ),
            all(
                target_os = "netbsd",
                any(target_arch = "aarch64", target_arch = "arm", target_arch = "powerpc")
            ),
            all(
                target_os = "vxworks",
                any(
                    target_arch = "aarch64",
                    target_arch = "arm",
                    target_arch = "powerpc64",
                    target_arch = "powerpc"
                )
            ),
            all(
                target_os = "fuchsia",
                any(target_arch = "aarch64", target_arch = "riscv64")
            ),
            all(target_os = "nto", target_arch = "aarch64"),
            target_os = "horizon"
        ))] {
            pub type c_char = u8;
            pub type NonZero_c_char = crate::num::NonZeroU8;
        } else {
            // On every other target, c_char is signed.
            pub type c_char = i8;
            pub type NonZero_c_char = crate::num::NonZeroI8;
        }
    }
}

mod c_int_definition {
    cfg_if! {
        if #[cfg(any(target_arch = "avr", target_arch = "msp430"))] {
            pub type c_int = i16;
            pub type NonZero_c_int = crate::num::NonZeroI16;
            pub type c_uint = u16;
            pub type NonZero_c_uint = crate::num::NonZeroU16;
        } else {
            pub type c_int = i32;
            pub type NonZero_c_int = crate::num::NonZeroI32;
            pub type c_uint = u32;
            pub type NonZero_c_uint = crate::num::NonZeroU32;
        }
    }
}

mod c_long_definition {
    cfg_if! {
        if #[cfg(all(target_pointer_width = "64", not(windows)))] {
            pub type c_long = i64;
            pub type NonZero_c_long = crate::num::NonZeroI64;
            pub type c_ulong = u64;
            pub type NonZero_c_ulong = crate::num::NonZeroU64;
        } else {
            // The minimal size of `long` in the C standard is 32 bits
            pub type c_long = i32;
            pub type NonZero_c_long = crate::num::NonZeroI32;
            pub type c_ulong = u32;
            pub type NonZero_c_ulong = crate::num::NonZeroU32;
        }
    }
}

// N.B., for LLVM to recognize the void pointer type and by extension
//     functions like malloc(), we need to have it represented as i8* in
//     LLVM bitcode. The enum used here ensures this and prevents misuse
//     of the "raw" type by only having private variants. We need two
//     variants, because the compiler complains about the repr attribute
//     otherwise and we need at least one variant as otherwise the enum
//     would be uninhabited and at least dereferencing such pointers would
//     be UB.
#[doc = include_str!("c_void.md")]
#[lang = "c_void"]
#[cfg_attr(not(doc), repr(u8))] // work around https://github.com/rust-lang/rust/issues/90435
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
        f.debug_struct("c_void").finish()
    }
}

/// Basic implementation of a `va_list`.
// The name is WIP, using `VaListImpl` for now.
#[cfg(any(
    all(
        not(target_arch = "aarch64"),
        not(target_arch = "powerpc"),
        not(target_arch = "s390x"),
        not(target_arch = "x86_64")
    ),
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")),
    target_family = "wasm",
    target_arch = "asmjs",
    target_os = "uefi",
    windows,
))]
#[cfg_attr(not(doc), repr(transparent))] // work around https://github.com/rust-lang/rust/issues/90435
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
    all(
        not(target_arch = "aarch64"),
        not(target_arch = "powerpc"),
        not(target_arch = "s390x"),
        not(target_arch = "x86_64")
    ),
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")),
    target_family = "wasm",
    target_arch = "asmjs",
    target_os = "uefi",
    windows,
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
    not(target_os = "uefi"),
    not(windows),
))]
#[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
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
#[cfg(all(target_arch = "powerpc", not(target_os = "uefi"), not(windows)))]
#[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
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

/// s390x ABI implementation of a `va_list`.
#[cfg(target_arch = "s390x")]
#[cfg_attr(not(doc), repr(C))] // work around https://github.com/rust-lang/rust/issues/66401
#[derive(Debug)]
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
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
#[cfg_attr(not(doc), repr(transparent))] // work around https://github.com/rust-lang/rust/issues/90435
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
            not(target_arch = "s390x"),
            not(target_arch = "x86_64")
        ),
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")),
        target_family = "wasm",
        target_arch = "asmjs",
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
        any(not(target_arch = "aarch64"), not(any(target_os = "macos", target_os = "ios"))),
        not(target_family = "wasm"),
        not(target_arch = "asmjs"),
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
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")),
    target_family = "wasm",
    target_arch = "asmjs",
    target_os = "uefi",
    windows,
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
    any(
        target_arch = "aarch64",
        target_arch = "powerpc",
        target_arch = "s390x",
        target_arch = "x86_64"
    ),
    any(not(target_arch = "aarch64"), not(any(target_os = "macos", target_os = "ios"))),
    not(target_family = "wasm"),
    not(target_arch = "asmjs"),
    not(target_os = "uefi"),
    not(windows),
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
    #[rustc_nounwind]
    fn va_end(ap: &mut VaListImpl<'_>);

    /// Copies the current location of arglist `src` to the arglist `dst`.
    #[rustc_nounwind]
    fn va_copy<'f>(dest: *mut VaListImpl<'f>, src: &VaListImpl<'f>);

    /// Loads an argument of type `T` from the `va_list` `ap` and increment the
    /// argument `ap` points to.
    #[rustc_nounwind]
    fn va_arg<T: sealed_trait::VaArgSafe>(ap: &mut VaListImpl<'_>) -> T;
}
