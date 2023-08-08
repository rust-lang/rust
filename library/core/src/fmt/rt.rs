#![allow(missing_debug_implementations)]
#![unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]

//! These are the lang items used by format_args!().

use super::*;

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

impl Placeholder {
    #[inline(always)]
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
    Is(usize),
    /// Specified using `$` and `*` syntaxes, stores the index into `args`
    Param(usize),
    /// Not specified
    Implied,
}

// This needs to match the order of flags in compiler/rustc_ast_lowering/src/format.rs.
#[derive(Copy, Clone)]
pub(super) enum Flag {
    SignPlus,
    SignMinus,
    Alternate,
    SignAwareZeroPad,
    DebugLowerHex,
    DebugUpperHex,
}

/// This struct represents the generic "argument" which is taken by format_args!().
/// It contains a function to format the given value. At compile time it is ensured that the
/// function and the value have the correct types, and then this struct is used to canonicalize
/// arguments to one type.
///
/// Argument is essentially an optimized partially applied formatting function,
/// equivalent to `exists T.(&T, fn(&T, &mut Formatter<'_>) -> Result`.
#[lang = "format_argument"]
#[derive(Copy, Clone)]
pub struct Argument<'a> {
    value: &'a Opaque,
    formatter: fn(&Opaque, &mut Formatter<'_>) -> Result,
}

#[rustc_diagnostic_item = "ArgumentMethods"]
impl<'a> Argument<'a> {
    #[inline(always)]
    fn new<'b, T>(x: &'b T, f: fn(&T, &mut Formatter<'_>) -> Result) -> Argument<'b> {
        // SAFETY: `mem::transmute(x)` is safe because
        //     1. `&'b T` keeps the lifetime it originated with `'b`
        //              (so as to not have an unbounded lifetime)
        //     2. `&'b T` and `&'b Opaque` have the same memory layout
        //              (when `T` is `Sized`, as it is here)
        // `mem::transmute(f)` is safe since `fn(&T, &mut Formatter<'_>) -> Result`
        // and `fn(&Opaque, &mut Formatter<'_>) -> Result` have the same ABI
        // (as long as `T` is `Sized`)
        unsafe { Argument { formatter: mem::transmute(f), value: mem::transmute(x) } }
    }

    #[inline(always)]
    pub fn new_display<'b, T: Display>(x: &'b T) -> Argument<'_> {
        Self::new(x, Display::fmt)
    }
    #[inline(always)]
    pub fn new_debug<'b, T: Debug>(x: &'b T) -> Argument<'_> {
        Self::new(x, Debug::fmt)
    }
    #[inline(always)]
    pub fn new_octal<'b, T: Octal>(x: &'b T) -> Argument<'_> {
        Self::new(x, Octal::fmt)
    }
    #[inline(always)]
    pub fn new_lower_hex<'b, T: LowerHex>(x: &'b T) -> Argument<'_> {
        Self::new(x, LowerHex::fmt)
    }
    #[inline(always)]
    pub fn new_upper_hex<'b, T: UpperHex>(x: &'b T) -> Argument<'_> {
        Self::new(x, UpperHex::fmt)
    }
    #[inline(always)]
    pub fn new_pointer<'b, T: Pointer>(x: &'b T) -> Argument<'_> {
        Self::new(x, Pointer::fmt)
    }
    #[inline(always)]
    pub fn new_binary<'b, T: Binary>(x: &'b T) -> Argument<'_> {
        Self::new(x, Binary::fmt)
    }
    #[inline(always)]
    pub fn new_lower_exp<'b, T: LowerExp>(x: &'b T) -> Argument<'_> {
        Self::new(x, LowerExp::fmt)
    }
    #[inline(always)]
    pub fn new_upper_exp<'b, T: UpperExp>(x: &'b T) -> Argument<'_> {
        Self::new(x, UpperExp::fmt)
    }
    #[inline(always)]
    pub fn from_usize(x: &usize) -> Argument<'_> {
        Self::new(x, USIZE_MARKER)
    }

    #[inline(always)]
    pub(super) fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        (self.formatter)(self.value, f)
    }

    #[inline(always)]
    pub(super) fn as_usize(&self) -> Option<usize> {
        // We are type punning a bit here: USIZE_MARKER only takes an &usize but
        // formatter takes an &Opaque. Rust understandably doesn't think we should compare
        // the function pointers if they don't have the same signature, so we cast to
        // usizes to tell it that we just want to compare addresses.
        if self.formatter as usize == USIZE_MARKER as usize {
            // SAFETY: The `formatter` field is only set to USIZE_MARKER if
            // the value is a usize, so this is safe
            Some(unsafe { *(self.value as *const _ as *const usize) })
        } else {
            None
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
    #[inline(always)]
    pub fn none() -> [Self; 0] {
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
    #[inline(always)]
    pub unsafe fn new() -> Self {
        Self { _private: () }
    }
}

extern "C" {
    type Opaque;
}

// This guarantees a single stable value for the function pointer associated with
// indices/counts in the formatting infrastructure.
//
// Note that a function defined as such would not be correct as functions are
// always tagged unnamed_addr with the current lowering to LLVM IR, so their
// address is not considered important to LLVM and as such the as_usize cast
// could have been miscompiled. In practice, we never call as_usize on non-usize
// containing data (as a matter of static generation of the formatting
// arguments), so this is merely an additional check.
//
// We primarily want to ensure that the function pointer at `USIZE_MARKER` has
// an address corresponding *only* to functions that also take `&usize` as their
// first argument. The read_volatile here ensures that we can safely ready out a
// usize from the passed reference and that this address does not point at a
// non-usize taking function.
static USIZE_MARKER: fn(&usize, &mut Formatter<'_>) -> Result = |ptr, _| {
    // SAFETY: ptr is a reference
    let _v: usize = unsafe { crate::ptr::read_volatile(ptr) };
    loop {}
};
