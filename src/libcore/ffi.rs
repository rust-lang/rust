#![stable(feature = "", since = "1.30.0")]

#![allow(non_camel_case_types)]

//! Utilities related to FFI bindings.

use crate::fmt;

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
/// [pointer]: ../../std/primitive.pointer.html
/// [Nomicon]: https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
// N.B., for LLVM to recognize the void pointer type and by extension
//     functions like malloc(), we need to have it represented as i8* in
//     LLVM bitcode. The enum used here ensures this and prevents misuse
//     of the "raw" type by only having private variants. We need two
//     variants, because the compiler complains about the repr attribute
//     otherwise and we need at least one variant as otherwise the enum
//     would be uninhabited and at least dereferencing such pointers would
//     be UB.
#[repr(u8)]
#[stable(feature = "raw_os", since = "1.1.0")]
pub enum c_void {
    #[unstable(feature = "c_void_variant", reason = "temporary implementation detail",
               issue = "0")]
    #[doc(hidden)] __variant1,
    #[unstable(feature = "c_void_variant", reason = "temporary implementation detail",
               issue = "0")]
    #[doc(hidden)] __variant2,
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for c_void {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("c_void")
    }
}

/// Basic implementation of a `va_list`.
#[cfg(any(all(not(target_arch = "aarch64"), not(target_arch = "powerpc"),
              not(target_arch = "x86_64")),
          all(target_arch = "aarch64", target_os = "ios"),
          windows))]
#[unstable(feature = "c_variadic",
           reason = "the `c_variadic` feature has not been properly tested on \
                     all supported platforms",
           issue = "44930")]
extern {
    type VaListImpl;
}

#[cfg(any(all(not(target_arch = "aarch64"), not(target_arch = "powerpc"),
              not(target_arch = "x86_64")),
          all(target_arch = "aarch64", target_os = "ios"),
          windows))]
impl fmt::Debug for VaListImpl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "va_list* {:p}", self)
    }
}

/// AArch64 ABI implementation of a `va_list`. See the
/// [AArch64 Procedure Call Standard] for more details.
///
/// [AArch64 Procedure Call Standard]:
/// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf
#[cfg(all(target_arch = "aarch64", not(target_os = "ios"), not(windows)))]
#[repr(C)]
#[derive(Debug)]
#[unstable(feature = "c_variadic",
           reason = "the `c_variadic` feature has not been properly tested on \
                     all supported platforms",
           issue = "44930")]
struct VaListImpl {
    stack: *mut c_void,
    gr_top: *mut c_void,
    vr_top: *mut c_void,
    gr_offs: i32,
    vr_offs: i32,
}

/// PowerPC ABI implementation of a `va_list`.
#[cfg(all(target_arch = "powerpc", not(windows)))]
#[repr(C)]
#[derive(Debug)]
#[unstable(feature = "c_variadic",
           reason = "the `c_variadic` feature has not been properly tested on \
                     all supported platforms",
           issue = "44930")]
struct VaListImpl {
    gpr: u8,
    fpr: u8,
    reserved: u16,
    overflow_arg_area: *mut c_void,
    reg_save_area: *mut c_void,
}

/// x86_64 ABI implementation of a `va_list`.
#[cfg(all(target_arch = "x86_64", not(windows)))]
#[repr(C)]
#[derive(Debug)]
#[unstable(feature = "c_variadic",
           reason = "the `c_variadic` feature has not been properly tested on \
                     all supported platforms",
           issue = "44930")]
struct VaListImpl {
    gp_offset: i32,
    fp_offset: i32,
    overflow_arg_area: *mut c_void,
    reg_save_area: *mut c_void,
}

/// A wrapper for a `va_list`
#[lang = "va_list"]
#[derive(Debug)]
#[unstable(feature = "c_variadic",
           reason = "the `c_variadic` feature has not been properly tested on \
                     all supported platforms",
           issue = "44930")]
#[repr(transparent)]
pub struct VaList<'a>(&'a mut VaListImpl);

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
    /// Trait which whitelists the allowed types to be used with [VaList::arg]
    ///
    /// [VaList::va_arg]: struct.VaList.html#method.arg
    #[unstable(feature = "c_variadic",
               reason = "the `c_variadic` feature has not been properly tested on \
                         all supported platforms",
               issue = "44930")]
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

impl_va_arg_safe!{i8, i16, i32, i64, usize}
impl_va_arg_safe!{u8, u16, u32, u64, isize}
impl_va_arg_safe!{f64}

#[unstable(feature = "c_variadic",
           reason = "the `c_variadic` feature has not been properly tested on \
                     all supported platforms",
           issue = "44930")]
impl<T> sealed_trait::VaArgSafe for *mut T {}
#[unstable(feature = "c_variadic",
           reason = "the `c_variadic` feature has not been properly tested on \
                     all supported platforms",
           issue = "44930")]
impl<T> sealed_trait::VaArgSafe for *const T {}

impl<'a> VaList<'a> {
    /// Advance to the next arg.
    #[unstable(feature = "c_variadic",
               reason = "the `c_variadic` feature has not been properly tested on \
                         all supported platforms",
               issue = "44930")]
    pub unsafe fn arg<T: sealed_trait::VaArgSafe>(&mut self) -> T {
        va_arg(self)
    }

    /// Copies the `va_list` at the current location.
    #[unstable(feature = "c_variadic",
               reason = "the `c_variadic` feature has not been properly tested on \
                         all supported platforms",
               issue = "44930")]
    pub unsafe fn with_copy<F, R>(&self, f: F) -> R
            where F: for<'copy> FnOnce(VaList<'copy>) -> R {
        #[cfg(any(all(not(target_arch = "aarch64"), not(target_arch = "powerpc"),
                      not(target_arch = "x86_64")),
                  all(target_arch = "aarch64", target_os = "ios"),
                  windows))]
        let mut ap = va_copy(self);
        #[cfg(all(any(target_arch = "aarch64", target_arch = "powerpc", target_arch = "x86_64"),
                  not(windows), not(all(target_arch = "aarch64", target_os = "ios"))))]
        let mut ap_inner = va_copy(self);
        #[cfg(all(any(target_arch = "aarch64", target_arch = "powerpc", target_arch = "x86_64"),
                  not(windows), not(all(target_arch = "aarch64", target_os = "ios"))))]
        let mut ap = VaList(&mut ap_inner);
        let ret = f(VaList(ap.0));
        va_end(&mut ap);
        ret
    }
}

extern "rust-intrinsic" {
    /// Destroy the arglist `ap` after initialization with `va_start` or
    /// `va_copy`.
    fn va_end(ap: &mut VaList);

    /// Copies the current location of arglist `src` to the arglist `dst`.
    #[cfg(any(all(not(target_arch = "aarch64"), not(target_arch = "powerpc"),
                  not(target_arch = "x86_64")),
              all(target_arch = "aarch64", target_os = "ios"),
              windows))]
    fn va_copy<'a>(src: &VaList<'a>) -> VaList<'a>;
    #[cfg(all(any(target_arch = "aarch64", target_arch = "powerpc", target_arch = "x86_64"),
              not(windows), not(all(target_arch = "aarch64", target_os = "ios"))))]
    fn va_copy(src: &VaList) -> VaListImpl;

    /// Loads an argument of type `T` from the `va_list` `ap` and increment the
    /// argument `ap` points to.
    fn va_arg<T: sealed_trait::VaArgSafe>(ap: &mut VaList) -> T;
}
