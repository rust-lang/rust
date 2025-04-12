#![allow(missing_debug_implementations)]
#![unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]

//! These are the lang items used by format_args!().

use super::*;
use crate::hint::unreachable_unchecked;

#[lang = "format_placeholder"]
#[derive(Copy, Clone)]
pub struct Placeholder {
    pub position: usize,
    pub flags: u32,
    pub precision: Count,
    pub width: Count,
}

#[cfg(bootstrap)]
impl Placeholder {
    #[inline]
    pub const fn new(position: usize, flags: u32, precision: Count, width: Count) -> Self {
        Self { position, flags, precision, width }
    }
}

/// Used by [width](https://doc.rust-lang.org/std/fmt/#width)
/// and [precision](https://doc.rust-lang.org/std/fmt/#precision) specifiers.
#[lang = "format_count"]
#[derive(Copy, Clone)]
pub enum Count {
    /// Specified with a literal number, stores the value
    Is(u16),
    /// Specified using `$` and `*` syntaxes, stores the index into `args`
    Param(usize),
    /// Not specified
    Implied,
}

trait FormatThunk {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

#[derive(Copy, Clone)]
enum ArgumentType<'a> {
    Placeholder(&'a dyn FormatThunk),
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

macro_rules! implement_argument_constructor {
    ($trait:ident, $operand:expr, $fmt:item) => {{
        #[repr(transparent)]
        struct Wrapper<T>(T);

        // SAFETY: `Wrapper<T>` has the same memory layout as `T` due to #[repr(transparent)].
        let thunk = unsafe { mem::transmute::<&T, &Wrapper<T>>($operand) };

        impl<T: $trait> FormatThunk for Wrapper<T> {
            #[inline]
            $fmt
        }

        Self::new(thunk)
    }};

    ($trait:ident, $operand:expr) => {
        implement_argument_constructor!(
            $trait,
            $operand,
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let Self(inner) = self;
                inner.fmt(f)
            }
        )
    };
}

#[rustc_diagnostic_item = "ArgumentMethods"]
impl Argument<'_> {
    #[inline]
    fn new<'a>(x: &'a dyn FormatThunk) -> Argument<'a> {
        Argument { ty: ArgumentType::Placeholder(x) }
    }

    #[inline]
    pub fn new_display<T: Display>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(Display, x)
    }
    #[inline]
    pub fn new_debug<T: Debug>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(Debug, x)
    }
    #[inline]
    pub fn new_debug_noop<T: Debug>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(
            Debug,
            x,
            fn fmt(&self, _: &mut Formatter<'_>) -> Result {
                Ok(())
            }
        )
    }
    #[inline]
    pub fn new_octal<T: Octal>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(Octal, x)
    }
    #[inline]
    pub fn new_lower_hex<T: LowerHex>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(LowerHex, x)
    }
    #[inline]
    pub fn new_upper_hex<T: UpperHex>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(UpperHex, x)
    }
    #[inline]
    pub fn new_pointer<T: Pointer>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(Pointer, x)
    }
    #[inline]
    pub fn new_binary<T: Binary>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(Binary, x)
    }
    #[inline]
    pub fn new_lower_exp<T: LowerExp>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(LowerExp, x)
    }
    #[inline]
    pub fn new_upper_exp<T: UpperExp>(x: &T) -> Argument<'_> {
        implement_argument_constructor!(UpperExp, x)
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
            ArgumentType::Placeholder(thunk) => thunk.fmt(f),
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
