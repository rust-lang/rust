#![allow(missing_debug_implementations)]
#![unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]

//! These are the lang items used by format_args!().

use super::*;
use crate::hint::unreachable_unchecked;
use crate::ptr::NonNull;

#[lang = "format_placeholder"]
#[derive(Copy, Clone)]
pub struct Placeholder {
    pub position: usize,
    #[cfg(bootstrap)]
    pub fill: char,
    #[cfg(bootstrap)]
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
#[lang = "format_count"]
#[derive(Copy, Clone)]
pub enum Count {
    /// Specified with a literal number, stores the value
    #[cfg(bootstrap)]
    Is(usize),
    /// Specified with a literal number, stores the value
    #[cfg(not(bootstrap))]
    Is(u16),
    /// Specified using `$` and `*` syntaxes, stores the index into `args`
    Param(usize),
    /// Not specified
    Implied,
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

/// This struct represents the unsafety of constructing an `Arguments`.
/// It exists, rather than an unsafe function, in order to simplify the expansion
/// of `format_args!(..)` and reduce the scope of the `unsafe` block.
#[lang = "format_unsafe_arg"]
pub struct UnsafeArg {
    _private: (),
}

impl UnsafeArg {
    /// See documentation where `UnsafeArg` is required to know when it is safe to
    /// create and use `UnsafeArg`.
    #[inline]
    pub const unsafe fn new() -> Self {
        Self { _private: () }
    }
}
