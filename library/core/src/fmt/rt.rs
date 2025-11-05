#![allow(missing_debug_implementations)]
#![unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]

//! All types and methods in this file are used by the compiler in
//! the expansion/lowering of format_args!().
//!
//! Do not modify them without understanding the consequences for the format_args!() macro.

use super::*;
use crate::hint::unreachable_unchecked;
use crate::marker::PhantomData;
use crate::ptr::NonNull;

#[lang = "format_template"]
#[derive(Copy, Clone)]
pub struct Template<'a> {
    pub(super) pieces: NonNull<rt::Piece>,
    lifetime: PhantomData<&'a rt::Piece>,
}

unsafe impl Send for Template<'_> {}
unsafe impl Sync for Template<'_> {}

impl<'a> Template<'a> {
    #[inline]
    pub const unsafe fn new<const N: usize>(pieces: &'a [rt::Piece; N]) -> Self {
        Self { pieces: NonNull::from_ref(pieces).cast(), lifetime: PhantomData }
    }

    #[inline]
    pub const unsafe fn next(&mut self) -> Piece {
        // SAFETY: Guaranteed by caller.
        unsafe {
            let piece = *self.pieces.as_ref();
            self.pieces = self.pieces.add(1);
            piece
        }
    }
}

#[lang = "format_piece"]
#[derive(Copy, Clone)]
pub union Piece {
    pub i: usize,
    pub p: *const u8,
}

unsafe impl Send for Piece {}
unsafe impl Sync for Piece {}

// These are marked as #[stable] because of #[rustc_promotable] and #[rustc_const_stable].
// With #[rustc_const_unstable], many format_args!() invocations would result in errors.
//
// There is still no way to use these on stable, because Piece itself is #[unstable] and not
// reachable through any public path. (format_args!()'s expansion uses it as a lang item.)
impl Piece {
    #[rustc_promotable]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "rust1", since = "1.0.0")]
    pub const fn str(s: &'static str) -> Self {
        Self { p: s as *const str as *const u8 }
    }

    #[rustc_promotable]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "rust1", since = "1.0.0")]
    pub const fn num(i: usize) -> Self {
        Self { i }
    }
}

#[derive(Copy, Clone)]
enum ArgumentType<'a> {
    Placeholder {
        // INVARIANT: `formatter` has type `fn(&T, _) -> _` for some `T`, and `value`
        // was derived from a `&'a T`.
        value: NonNull<()>,
        formatter: unsafe fn(NonNull<()>, &mut Formatter<'_>) -> Result,
        _lifetime: PhantomData<&'a ()>,
    },
    Count(u16),
}

/// This struct represents a generic "argument" which is taken by format_args!().
///
/// This can be either a placeholder argument or a count argument.
/// * A placeholder argument contains a function to format the given value. At
///   compile time it is ensured that the function and the value have the correct
///   types, and then this struct is used to canonicalize arguments to one type.
///   Placeholder arguments are essentially an optimized partially applied formatting
///   function, equivalent to `exists T.(&T, fn(&T, &mut Formatter<'_>) -> Result`.
/// * A count argument contains a count for dynamic formatting parameters like
///   precision and width.
#[lang = "format_argument"]
#[derive(Copy, Clone)]
pub struct Argument<'a> {
    ty: ArgumentType<'a>,
}

macro_rules! argument_new {
    ($t:ty, $x:expr, $f:expr) => {
        Argument {
            // INVARIANT: this creates an `ArgumentType<'a>` from a `&'a T` and
            // a `fn(&T, ...)`, so the invariant is maintained.
            ty: ArgumentType::Placeholder {
                value: NonNull::<$t>::from_ref($x).cast(),
                // The Rust ABI considers all pointers to be equivalent, so transmuting a fn(&T) to
                // fn(NonNull<()>) and calling it with a NonNull<()> that points at a T is allowed.
                // However, the CFI sanitizer does not allow this, and triggers a crash when it
                // happens.
                //
                // To avoid this crash, we use a helper function when CFI is enabled. To avoid the
                // cost of this helper function (mainly code-size) when it is not needed, we
                // transmute the function pointer otherwise.
                //
                // This is similar to what the Rust compiler does internally with vtables when KCFI
                // is enabled, where it generates trampoline functions that only serve to adjust the
                // expected type of the argument. `ArgumentType::Placeholder` is a bit like a
                // manually constructed trait object, so it is not surprising that the same approach
                // has to be applied here as well.
                //
                // It is still considered problematic (from the Rust side) that CFI rejects entirely
                // legal Rust programs, so we do not consider anything done here a stable guarantee,
                // but meanwhile we carry this work-around to keep Rust compatible with CFI and
                // KCFI.
                #[cfg(not(any(sanitize = "cfi", sanitize = "kcfi")))]
                formatter: {
                    let f: fn(&$t, &mut Formatter<'_>) -> Result = $f;
                    // SAFETY: This is only called with `value`, which has the right type.
                    unsafe { core::mem::transmute(f) }
                },
                #[cfg(any(sanitize = "cfi", sanitize = "kcfi"))]
                formatter: |ptr: NonNull<()>, fmt: &mut Formatter<'_>| {
                    let func = $f;
                    // SAFETY: This is the same type as the `value` field.
                    let r = unsafe { ptr.cast::<$t>().as_ref() };
                    (func)(r, fmt)
                },
                _lifetime: PhantomData,
            },
        }
    };
}

impl Argument<'_> {
    #[inline]
    pub const fn new_display<T: Display>(x: &T) -> Argument<'_> {
        argument_new!(T, x, <T as Display>::fmt)
    }
    #[inline]
    pub const fn new_debug<T: Debug>(x: &T) -> Argument<'_> {
        argument_new!(T, x, <T as Debug>::fmt)
    }
    #[inline]
    pub const fn new_debug_noop<T: Debug>(x: &T) -> Argument<'_> {
        argument_new!(T, x, |_: &T, _| Ok(()))
    }
    #[inline]
    pub const fn new_octal<T: Octal>(x: &T) -> Argument<'_> {
        argument_new!(T, x, <T as Octal>::fmt)
    }
    #[inline]
    pub const fn new_lower_hex<T: LowerHex>(x: &T) -> Argument<'_> {
        argument_new!(T, x, <T as LowerHex>::fmt)
    }
    #[inline]
    pub const fn new_upper_hex<T: UpperHex>(x: &T) -> Argument<'_> {
        argument_new!(T, x, <T as UpperHex>::fmt)
    }
    #[inline]
    pub const fn new_pointer<T: Pointer>(x: &T) -> Argument<'_> {
        argument_new!(T, x, <T as Pointer>::fmt)
    }
    #[inline]
    pub const fn new_binary<T: Binary>(x: &T) -> Argument<'_> {
        argument_new!(T, x, <T as Binary>::fmt)
    }
    #[inline]
    pub const fn new_lower_exp<T: LowerExp>(x: &T) -> Argument<'_> {
        argument_new!(T, x, <T as LowerExp>::fmt)
    }
    #[inline]
    pub const fn new_upper_exp<T: UpperExp>(x: &T) -> Argument<'_> {
        argument_new!(T, x, <T as UpperExp>::fmt)
    }
    #[inline]
    #[track_caller]
    pub const fn from_usize(x: &usize) -> Argument<'_> {
        if *x > u16::MAX as usize {
            panic!("Formatting argument out of range");
        }
        Argument { ty: ArgumentType::Count(*x as u16) }
    }

    /// Format this placeholder argument.
    ///
    /// # Safety
    ///
    /// This argument must actually be a placeholder argument.
    #[inline]
    pub(super) unsafe fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.ty {
            // SAFETY:
            // Because of the invariant that if `formatter` had the type
            // `fn(&T, _) -> _` then `value` has type `&'b T` where `'b` is
            // the lifetime of the `ArgumentType`, and because references
            // and `NonNull` are ABI-compatible, this is completely equivalent
            // to calling the original function passed to `new` with the
            // original reference, which is sound.
            ArgumentType::Placeholder { formatter, value, .. } => unsafe { formatter(value, f) },
            // SAFETY: the caller promised this.
            ArgumentType::Count(_) => unsafe { unreachable_unchecked() },
        }
    }

    #[inline]
    pub(super) const fn as_u16(&self) -> Option<u16> {
        match self.ty {
            ArgumentType::Count(count) => Some(count),
            ArgumentType::Placeholder { .. } => None,
        }
    }
}

/// Used by the format_args!() macro to create a fmt::Arguments object.
#[doc(hidden)]
#[rustc_diagnostic_item = "FmtArgumentsNew"]
impl<'a> Arguments<'a> {
    #[inline]
    pub const fn new_const(template: rt::Template<'a>) -> Arguments<'a> {
        Arguments { template, args: NonNull::dangling() }
    }

    #[inline]
    pub fn new<const N: usize>(
        template: rt::Template<'a>,
        args: &'a [rt::Argument<'a>; N],
    ) -> Arguments<'a> {
        Arguments { template, args: NonNull::from_ref(args).cast() }
    }

    // These two methods are used in library/core/src/panicking.rs to create a
    // `fmt::Arguments` for a `&'static str`.
    #[inline]
    pub(crate) const fn pieces_for_str(s: &'static str) -> [rt::Piece; 3] {
        [rt::Piece { i: s.len() as _ }, rt::Piece::str(s), rt::Piece { i: 0 }]
    }
    /// Safety: Only call this with the result of `pieces_for_str`.
    #[inline]
    pub(crate) const unsafe fn from_pieces(p: &'a [rt::Piece; 3]) -> Self {
        // SAFETY: Guaranteed by caller.
        Self::new_const(unsafe { rt::Template::new(p) })
    }
}
