#![allow(missing_debug_implementations)]
#![unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]

//! These are the lang items used by format_args!().

use super::*;
use crate::hint::unreachable_unchecked;
use crate::marker::PhantomData;
use crate::ptr::NonNull;

#[cfg(not(bootstrap))]
#[lang = "format_template"]
#[derive(Copy, Clone)]
pub struct Template<'a> {
    pub(super) pieces: NonNull<rt::Piece>,
    lifetime: PhantomData<&'a rt::Piece>,
}

#[cfg(not(bootstrap))]
unsafe impl Send for Template<'_> {}
#[cfg(not(bootstrap))]
unsafe impl Sync for Template<'_> {}

#[cfg(not(bootstrap))]
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

#[cfg(not(bootstrap))]
#[lang = "format_piece"]
#[derive(Copy, Clone)]
pub union Piece {
    pub i: usize,
    pub p: *const u8,
}

#[cfg(not(bootstrap))]
unsafe impl Send for Piece {}
#[cfg(not(bootstrap))]
unsafe impl Sync for Piece {}

// These are marked as #[stable] because of #[rustc_promotable] and #[rustc_const_stable].
// With #[rustc_const_unstable], many format_args!() invocations would result in errors.
//
// There is still no way to use these on stable, because Piece itself is #[unstable] and not
// reachable through any public path. (format_args!()'s expansion uses it as a lang item.)
#[cfg(not(bootstrap))]
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

#[cfg(bootstrap)]
#[lang = "format_placeholder"]
#[derive(Copy, Clone)]
pub struct Placeholder {
    pub position: usize,
    pub fill: char,
    pub align: Alignment,
    pub flags: u32,
    pub precision: Count,
    pub width: Count,
}

#[cfg(bootstrap)]
impl Placeholder {
    #[inline]
    pub const fn new(
        position: usize,
        fill: char,
        align: Alignment,
        flags: u32,
        precision: Count,
        width: Count,
    ) -> Self {
        Self { position, fill, align, flags, precision, width }
    }
}

#[cfg(bootstrap)]
#[lang = "format_alignment"]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Right,
    Center,
    Unknown,
}

/// Used by [width](https://doc.rust-lang.org/std/fmt/#width)
/// and [precision](https://doc.rust-lang.org/std/fmt/#precision) specifiers.
#[cfg(bootstrap)]
#[lang = "format_count"]
#[derive(Copy, Clone)]
pub enum Count {
    /// Specified with a literal number, stores the value
    Is(usize),
    /// Specified using `$` and `*` syntaxes, stores the index into `args`
    Param(usize),
    /// Not specified
    Implied,
}

// This needs to match with compiler/rustc_ast_lowering/src/format.rs.
pub const SIGN_PLUS_FLAG: u32 = 1 << 21;
pub const SIGN_MINUS_FLAG: u32 = 1 << 22;
pub const ALTERNATE_FLAG: u32 = 1 << 23;
pub const SIGN_AWARE_ZERO_PAD_FLAG: u32 = 1 << 24;
pub const DEBUG_LOWER_HEX_FLAG: u32 = 1 << 25;
pub const DEBUG_UPPER_HEX_FLAG: u32 = 1 << 26;
pub const WIDTH_FLAG: u32 = 1 << 27;
pub const PRECISION_FLAG: u32 = 1 << 28;
pub const ALIGN_BITS: u32 = 0b11 << 29;
pub const ALIGN_LEFT: u32 = 0 << 29;
pub const ALIGN_RIGHT: u32 = 1 << 29;
pub const ALIGN_CENTER: u32 = 2 << 29;
pub const ALIGN_UNKNOWN: u32 = 3 << 29;
pub const ALWAYS_SET: u32 = 1 << 31;

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

#[rustc_diagnostic_item = "ArgumentMethods"]
impl Argument<'_> {
    #[inline]
    const fn new<'a, T>(x: &'a T, f: fn(&T, &mut Formatter<'_>) -> Result) -> Argument<'a> {
        Argument {
            // INVARIANT: this creates an `ArgumentType<'a>` from a `&'a T` and
            // a `fn(&T, ...)`, so the invariant is maintained.
            ty: ArgumentType::Placeholder {
                value: NonNull::from_ref(x).cast(),
                // SAFETY: function pointers always have the same layout.
                formatter: unsafe { mem::transmute(f) },
                _lifetime: PhantomData,
            },
        }
    }

    #[inline]
    pub fn new_display<T: Display>(x: &T) -> Argument<'_> {
        Self::new(x, Display::fmt)
    }
    #[inline]
    pub fn new_debug<T: Debug>(x: &T) -> Argument<'_> {
        Self::new(x, Debug::fmt)
    }
    #[inline]
    pub fn new_debug_noop<T: Debug>(x: &T) -> Argument<'_> {
        Self::new(x, |_, _| Ok(()))
    }
    #[inline]
    pub fn new_octal<T: Octal>(x: &T) -> Argument<'_> {
        Self::new(x, Octal::fmt)
    }
    #[inline]
    pub fn new_lower_hex<T: LowerHex>(x: &T) -> Argument<'_> {
        Self::new(x, LowerHex::fmt)
    }
    #[inline]
    pub fn new_upper_hex<T: UpperHex>(x: &T) -> Argument<'_> {
        Self::new(x, UpperHex::fmt)
    }
    #[inline]
    pub fn new_pointer<T: Pointer>(x: &T) -> Argument<'_> {
        Self::new(x, Pointer::fmt)
    }
    #[inline]
    pub fn new_binary<T: Binary>(x: &T) -> Argument<'_> {
        Self::new(x, Binary::fmt)
    }
    #[inline]
    pub fn new_lower_exp<T: LowerExp>(x: &T) -> Argument<'_> {
        Self::new(x, LowerExp::fmt)
    }
    #[inline]
    pub fn new_upper_exp<T: UpperExp>(x: &T) -> Argument<'_> {
        Self::new(x, UpperExp::fmt)
    }
    #[inline]
    pub const fn from_usize(x: &usize) -> Argument<'_> {
        if *x > u16::MAX as usize {
            panic!("Formatting argument out of range");
        };
        Argument { ty: ArgumentType::Count(*x as u16) }
    }

    /// Format this placeholder argument.
    ///
    /// # Safety
    ///
    /// This argument must actually be a placeholder argument.
    ///
    // FIXME: Transmuting formatter in new and indirectly branching to/calling
    // it here is an explicit CFI violation.
    #[allow(inline_no_sanitize)]
    #[no_sanitize(cfi, kcfi)]
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

    /// Used by `format_args` when all arguments are gone after inlining,
    /// when using `&[]` would incorrectly allow for a bigger lifetime.
    ///
    /// This fails without format argument inlining, and that shouldn't be different
    /// when the argument is inlined:
    ///
    /// ```compile_fail,E0716
    /// let f = format_args!("{}", "a");
    /// println!("{f}");
    /// ```
    #[inline]
    pub const fn none() -> [Self; 0] {
        []
    }
}

#[cfg(bootstrap)]
#[lang = "format_unsafe_arg"]
pub struct UnsafeArg {
    _private: (),
}

#[cfg(bootstrap)]
impl UnsafeArg {
    #[inline]
    pub const unsafe fn new() -> Self {
        Self { _private: () }
    }
}
