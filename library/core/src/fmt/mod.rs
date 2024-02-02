//! Utilities for formatting and printing strings.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::cell::{Cell, Ref, RefCell, RefMut, SyncUnsafeCell, UnsafeCell};
use crate::char::{EscapeDebugExtArgs, MAX_LEN_UTF8};
use crate::marker::PhantomData;
use crate::num::fmt as numfmt;
use crate::ops::Deref;
use crate::{iter, mem, result, str};

mod builders;
#[cfg(not(no_fp_fmt_parse))]
mod float;
#[cfg(no_fp_fmt_parse)]
mod nofloat;
mod num;
mod rt;

#[stable(feature = "fmt_flags_align", since = "1.28.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "Alignment")]
/// Possible alignments returned by `Formatter::align`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Alignment {
    #[stable(feature = "fmt_flags_align", since = "1.28.0")]
    /// Indication that contents should be left-aligned.
    Left,
    #[stable(feature = "fmt_flags_align", since = "1.28.0")]
    /// Indication that contents should be right-aligned.
    Right,
    #[stable(feature = "fmt_flags_align", since = "1.28.0")]
    /// Indication that contents should be center-aligned.
    Center,
}

#[doc(hidden)]
#[unstable(feature = "fmt_internals", reason = "internal to standard library", issue = "none")]
impl From<rt::Alignment> for Option<Alignment> {
    fn from(value: rt::Alignment) -> Self {
        match value {
            rt::Alignment::Left => Some(Alignment::Left),
            rt::Alignment::Right => Some(Alignment::Right),
            rt::Alignment::Center => Some(Alignment::Center),
            rt::Alignment::Unknown => None,
        }
    }
}

#[stable(feature = "debug_builders", since = "1.2.0")]
pub use self::builders::{DebugList, DebugMap, DebugSet, DebugStruct, DebugTuple};
#[unstable(feature = "debug_closure_helpers", issue = "117729")]
pub use self::builders::{FromFn, from_fn};

/// The type returned by formatter methods.
///
/// # Examples
///
/// ```
/// use std::fmt;
///
/// #[derive(Debug)]
/// struct Triangle {
///     a: f32,
///     b: f32,
///     c: f32
/// }
///
/// impl fmt::Display for Triangle {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "({}, {}, {})", self.a, self.b, self.c)
///     }
/// }
///
/// let pythagorean_triple = Triangle { a: 3.0, b: 4.0, c: 5.0 };
///
/// assert_eq!(format!("{pythagorean_triple}"), "(3, 4, 5)");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub type Result = result::Result<(), Error>;

/// The error type which is returned from formatting a message into a stream.
///
/// This type does not support transmission of an error other than that an error
/// occurred. This is because, despite the existence of this error,
/// string formatting is considered an infallible operation.
/// `fmt()` implementors should not return this `Error` unless they received it from their
/// [`Formatter`]. The only time your code should create a new instance of this
/// error is when implementing `fmt::Write`, in order to cancel the formatting operation when
/// writing to the underlying stream fails.
///
/// Any extra information must be arranged to be transmitted through some other means,
/// such as storing it in a field to be consulted after the formatting operation has been
/// cancelled. (For example, this is how [`std::io::Write::write_fmt()`] propagates IO errors
/// during writing.)
///
/// This type, `fmt::Error`, should not be
/// confused with [`std::io::Error`] or [`std::error::Error`], which you may also
/// have in scope.
///
/// [`std::io::Error`]: ../../std/io/struct.Error.html
/// [`std::io::Write::write_fmt()`]: ../../std/io/trait.Write.html#method.write_fmt
/// [`std::error::Error`]: ../../std/error/trait.Error.html
///
/// # Examples
///
/// ```rust
/// use std::fmt::{self, write};
///
/// let mut output = String::new();
/// if let Err(fmt::Error) = write(&mut output, format_args!("Hello {}!", "world")) {
///     panic!("An error occurred");
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Error;

/// A trait for writing or formatting into Unicode-accepting buffers or streams.
///
/// This trait only accepts UTF-8–encoded data and is not [flushable]. If you only
/// want to accept Unicode and you don't need flushing, you should implement this trait;
/// otherwise you should implement [`std::io::Write`].
///
/// [`std::io::Write`]: ../../std/io/trait.Write.html
/// [flushable]: ../../std/io/trait.Write.html#tymethod.flush
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Write {
    /// Writes a string slice into this writer, returning whether the write
    /// succeeded.
    ///
    /// This method can only succeed if the entire string slice was successfully
    /// written, and this method will not return until all data has been
    /// written or an error occurs.
    ///
    /// # Errors
    ///
    /// This function will return an instance of [`std::fmt::Error`][Error] on error.
    ///
    /// The purpose of that error is to abort the formatting operation when the underlying
    /// destination encounters some error preventing it from accepting more text;
    /// in particular, it does not communicate any information about *what* error occurred.
    /// It should generally be propagated rather than handled, at least when implementing
    /// formatting traits.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::{Error, Write};
    ///
    /// fn writer<W: Write>(f: &mut W, s: &str) -> Result<(), Error> {
    ///     f.write_str(s)
    /// }
    ///
    /// let mut buf = String::new();
    /// writer(&mut buf, "hola")?;
    /// assert_eq!(&buf, "hola");
    /// # std::fmt::Result::Ok(())
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn write_str(&mut self, s: &str) -> Result;

    /// Writes a [`char`] into this writer, returning whether the write succeeded.
    ///
    /// A single [`char`] may be encoded as more than one byte.
    /// This method can only succeed if the entire byte sequence was successfully
    /// written, and this method will not return until all data has been
    /// written or an error occurs.
    ///
    /// # Errors
    ///
    /// This function will return an instance of [`Error`] on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::{Error, Write};
    ///
    /// fn writer<W: Write>(f: &mut W, c: char) -> Result<(), Error> {
    ///     f.write_char(c)
    /// }
    ///
    /// let mut buf = String::new();
    /// writer(&mut buf, 'a')?;
    /// writer(&mut buf, 'b')?;
    /// assert_eq!(&buf, "ab");
    /// # std::fmt::Result::Ok(())
    /// ```
    #[stable(feature = "fmt_write_char", since = "1.1.0")]
    fn write_char(&mut self, c: char) -> Result {
        self.write_str(c.encode_utf8(&mut [0; MAX_LEN_UTF8]))
    }

    /// Glue for usage of the [`write!`] macro with implementors of this trait.
    ///
    /// This method should generally not be invoked manually, but rather through
    /// the [`write!`] macro itself.
    ///
    /// # Errors
    ///
    /// This function will return an instance of [`Error`] on error. Please see
    /// [write_str](Write::write_str) for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::{Error, Write};
    ///
    /// fn writer<W: Write>(f: &mut W, s: &str) -> Result<(), Error> {
    ///     f.write_fmt(format_args!("{s}"))
    /// }
    ///
    /// let mut buf = String::new();
    /// writer(&mut buf, "world")?;
    /// assert_eq!(&buf, "world");
    /// # std::fmt::Result::Ok(())
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn write_fmt(&mut self, args: Arguments<'_>) -> Result {
        // We use a specialization for `Sized` types to avoid an indirection
        // through `&mut self`
        trait SpecWriteFmt {
            fn spec_write_fmt(self, args: Arguments<'_>) -> Result;
        }

        impl<W: Write + ?Sized> SpecWriteFmt for &mut W {
            #[inline]
            default fn spec_write_fmt(mut self, args: Arguments<'_>) -> Result {
                if let Some(s) = args.as_statically_known_str() {
                    self.write_str(s)
                } else {
                    write(&mut self, args)
                }
            }
        }

        impl<W: Write> SpecWriteFmt for &mut W {
            #[inline]
            fn spec_write_fmt(self, args: Arguments<'_>) -> Result {
                if let Some(s) = args.as_statically_known_str() {
                    self.write_str(s)
                } else {
                    write(self, args)
                }
            }
        }

        self.spec_write_fmt(args)
    }
}

#[stable(feature = "fmt_write_blanket_impl", since = "1.4.0")]
impl<W: Write + ?Sized> Write for &mut W {
    fn write_str(&mut self, s: &str) -> Result {
        (**self).write_str(s)
    }

    fn write_char(&mut self, c: char) -> Result {
        (**self).write_char(c)
    }

    fn write_fmt(&mut self, args: Arguments<'_>) -> Result {
        (**self).write_fmt(args)
    }
}

/// The signedness of a [`Formatter`] (or of a [`FormattingOptions`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[unstable(feature = "formatting_options", issue = "118117")]
pub enum Sign {
    /// Represents the `+` flag.
    Plus,
    /// Represents the `-` flag.
    Minus,
}

/// Specifies whether the [`Debug`] trait should use lower-/upper-case
/// hexadecimal or normal integers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[unstable(feature = "formatting_options", issue = "118117")]
pub enum DebugAsHex {
    /// Use lower-case hexadecimal integers for the `Debug` trait (like [the `x?` type](../../std/fmt/index.html#formatting-traits)).
    Lower,
    /// Use upper-case hexadecimal integers for the `Debug` trait (like [the `X?` type](../../std/fmt/index.html#formatting-traits)).
    Upper,
}

/// Options for formatting.
///
/// `FormattingOptions` is a [`Formatter`] without an attached [`Write`] trait.
/// It is mainly used to construct `Formatter` instances.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[unstable(feature = "formatting_options", issue = "118117")]
pub struct FormattingOptions {
    flags: u32,
    fill: char,
    align: Option<Alignment>,
    width: Option<usize>,
    precision: Option<usize>,
}

impl FormattingOptions {
    /// Construct a new `FormatterBuilder` with the supplied `Write` trait
    /// object for output that is equivalent to the `{}` formatting
    /// specifier:
    ///
    /// - no flags,
    /// - filled with spaces,
    /// - no alignment,
    /// - no width,
    /// - no precision, and
    /// - no [`DebugAsHex`] output mode.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn new() -> Self {
        Self { flags: 0, fill: ' ', align: None, width: None, precision: None }
    }

    /// Sets or removes the sign (the `+` or the `-` flag).
    ///
    /// - `+`: This is intended for numeric types and indicates that the sign
    /// should always be printed. By default only the negative sign of signed
    /// values is printed, and the sign of positive or unsigned values is
    /// omitted. This flag indicates that the correct sign (+ or -) should
    /// always be printed.
    /// - `-`: Currently not used
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn sign(&mut self, sign: Option<Sign>) -> &mut Self {
        self.flags =
            self.flags & !(1 << rt::Flag::SignMinus as u32 | 1 << rt::Flag::SignPlus as u32);
        match sign {
            None => {}
            Some(Sign::Plus) => self.flags |= 1 << rt::Flag::SignPlus as u32,
            Some(Sign::Minus) => self.flags |= 1 << rt::Flag::SignMinus as u32,
        }
        self
    }
    /// Sets or unsets the `0` flag.
    ///
    /// This is used to indicate for integer formats that the padding to width should both be done with a 0 character as well as be sign-aware
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn sign_aware_zero_pad(&mut self, sign_aware_zero_pad: bool) -> &mut Self {
        if sign_aware_zero_pad {
            self.flags |= 1 << rt::Flag::SignAwareZeroPad as u32
        } else {
            self.flags &= !(1 << rt::Flag::SignAwareZeroPad as u32)
        }
        self
    }
    /// Sets or unsets the `#` flag.
    ///
    /// This flag indicates that the "alternate" form of printing should be
    /// used. The alternate forms are:
    /// - [`Debug`] : pretty-print the [`Debug`] formatting (adds linebreaks and indentation)
    /// - [`LowerHex`] as well as [`UpperHex`] - precedes the argument with a `0x`
    /// - [`Octal`] - precedes the argument with a `0b`
    /// - [`Binary`] - precedes the argument with a `0o`
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn alternate(&mut self, alternate: bool) -> &mut Self {
        if alternate {
            self.flags |= 1 << rt::Flag::Alternate as u32
        } else {
            self.flags &= !(1 << rt::Flag::Alternate as u32)
        }
        self
    }
    /// Sets the fill character.
    ///
    /// The optional fill character and alignment is provided normally in
    /// conjunction with the width parameter. This indicates that if the value
    /// being formatted is smaller than width some extra characters will be
    /// printed around it.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn fill(&mut self, fill: char) -> &mut Self {
        self.fill = fill;
        self
    }
    /// Sets or removes the alignment.
    ///
    /// The alignment specifies how the value being formatted should be
    /// positioned if it is smaller than the width of the formatter.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn align(&mut self, align: Option<Alignment>) -> &mut Self {
        self.align = align;
        self
    }
    /// Sets or removes the width.
    ///
    /// This is a parameter for the “minimum width” that the format should take
    /// up. If the value’s string does not fill up this many characters, then
    /// the padding specified by [`FormattingOptions::fill`]/[`FormattingOptions::align`]
    /// will be used to take up the required space.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn width(&mut self, width: Option<usize>) -> &mut Self {
        self.width = width;
        self
    }
    /// Sets or removes the precision.
    ///
    /// - For non-numeric types, this can be considered a “maximum width”. If
    /// the resulting string is longer than this width, then it is truncated
    /// down to this many characters and that truncated value is emitted with
    /// proper fill, alignment and width if those parameters are set.
    /// - For integral types, this is ignored.
    /// - For floating-point types, this indicates how many digits after the
    /// decimal point should be printed.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn precision(&mut self, precision: Option<usize>) -> &mut Self {
        self.precision = precision;
        self
    }
    /// Specifies whether the [`Debug`] trait should use lower-/upper-case
    /// hexadecimal or normal integers
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn debug_as_hex(&mut self, debug_as_hex: Option<DebugAsHex>) -> &mut Self {
        self.flags = self.flags
            & !(1 << rt::Flag::DebugUpperHex as u32 | 1 << rt::Flag::DebugLowerHex as u32);
        match debug_as_hex {
            None => {}
            Some(DebugAsHex::Upper) => self.flags |= 1 << rt::Flag::DebugUpperHex as u32,
            Some(DebugAsHex::Lower) => self.flags |= 1 << rt::Flag::DebugLowerHex as u32,
        }
        self
    }

    /// Returns the current sign (the `+` or the `-` flag).
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn get_sign(&self) -> Option<Sign> {
        const SIGN_PLUS_BITFIELD: u32 = 1 << rt::Flag::SignPlus as u32;
        const SIGN_MINUS_BITFIELD: u32 = 1 << rt::Flag::SignMinus as u32;
        match self.flags & ((1 << rt::Flag::SignPlus as u32) | (1 << rt::Flag::SignMinus as u32)) {
            SIGN_PLUS_BITFIELD => Some(Sign::Plus),
            SIGN_MINUS_BITFIELD => Some(Sign::Minus),
            0 => None,
            _ => panic!("Invalid sign bits set in flags"),
        }
    }
    /// Returns the current `0` flag.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn get_sign_aware_zero_pad(&self) -> bool {
        self.flags & (1 << rt::Flag::SignAwareZeroPad as u32) != 0
    }
    /// Returns the current `#` flag.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn get_alternate(&self) -> bool {
        self.flags & (1 << rt::Flag::Alternate as u32) != 0
    }
    /// Returns the current fill character.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn get_fill(&self) -> char {
        self.fill
    }
    /// Returns the current alignment.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn get_align(&self) -> Option<Alignment> {
        self.align
    }
    /// Returns the current width.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn get_width(&self) -> Option<usize> {
        self.width
    }
    /// Returns the current precision.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn get_precision(&self) -> Option<usize> {
        self.precision
    }
    /// Returns the current precision.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn get_debug_as_hex(&self) -> Option<DebugAsHex> {
        const DEBUG_UPPER_BITFIELD: u32 = 1 << rt::Flag::DebugUpperHex as u32;
        const DEBUG_LOWER_BITFIELD: u32 = 1 << rt::Flag::DebugLowerHex as u32;
        match self.flags
            & ((1 << rt::Flag::DebugUpperHex as u32) | (1 << rt::Flag::DebugLowerHex as u32))
        {
            DEBUG_UPPER_BITFIELD => Some(DebugAsHex::Upper),
            DEBUG_LOWER_BITFIELD => Some(DebugAsHex::Lower),
            0 => None,
            _ => panic!("Invalid hex debug bits set in flags"),
        }
    }

    /// Creates a [`Formatter`] that writes its output to the given [`Write`] trait.
    ///
    /// You may alternatively use [`Formatter::new()`].
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn create_formatter<'a>(self, write: &'a mut (dyn Write + 'a)) -> Formatter<'a> {
        Formatter { options: self, buf: write }
    }

    #[doc(hidden)]
    #[unstable(
        feature = "fmt_internals",
        reason = "internal routines only exposed for testing",
        issue = "none"
    )]
    /// Flags for formatting
    pub fn flags(&mut self, flags: u32) {
        self.flags = flags
    }
    #[doc(hidden)]
    #[unstable(
        feature = "fmt_internals",
        reason = "internal routines only exposed for testing",
        issue = "none"
    )]
    /// Flags for formatting
    pub fn get_flags(&self) -> u32 {
        self.flags
    }
}

#[unstable(feature = "formatting_options", issue = "118117")]
impl Default for FormattingOptions {
    /// Same as [`FormattingOptions::new()`].
    fn default() -> Self {
        // The `#[derive(Default)]` implementation would set `fill` to `\0` instead of space.
        Self::new()
    }
}

/// Configuration for formatting.
///
/// A `Formatter` represents various options related to formatting. Users do not
/// construct `Formatter`s directly; a mutable reference to one is passed to
/// the `fmt` method of all formatting traits, like [`Debug`] and [`Display`].
///
/// To interact with a `Formatter`, you'll call various methods to change the
/// various options related to formatting. For examples, please see the
/// documentation of the methods defined on `Formatter` below.
#[allow(missing_debug_implementations)]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Formatter"]
pub struct Formatter<'a> {
    options: FormattingOptions,

    buf: &'a mut (dyn Write + 'a),
}

impl<'a> Formatter<'a> {
    /// Creates a new formatter with given [`FormattingOptions`].
    ///
    /// If `write` is a reference to a formatter, it is recommended to use
    /// [`Formatter::with_options`] instead as this can borrow the underlying
    /// `write`, thereby bypassing one layer of indirection.
    ///
    /// You may alternatively use [`FormattingOptions::create_formatter()`].
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn new(write: &'a mut (dyn Write + 'a), options: FormattingOptions) -> Self {
        Formatter { options, buf: write }
    }

    /// Creates a new formatter based on this one with given [`FormattingOptions`].
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub fn with_options<'b>(&'b mut self, options: FormattingOptions) -> Formatter<'b> {
        Formatter { options, buf: self.buf }
    }
}

/// This structure represents a safely precompiled version of a format string
/// and its arguments. This cannot be generated at runtime because it cannot
/// safely be done, so no constructors are given and the fields are private
/// to prevent modification.
///
/// The [`format_args!`] macro will safely create an instance of this structure.
/// The macro validates the format string at compile-time so usage of the
/// [`write()`] and [`format()`] functions can be safely performed.
///
/// You can use the `Arguments<'a>` that [`format_args!`] returns in `Debug`
/// and `Display` contexts as seen below. The example also shows that `Debug`
/// and `Display` format to the same thing: the interpolated format string
/// in `format_args!`.
///
/// ```rust
/// let debug = format!("{:?}", format_args!("{} foo {:?}", 1, 2));
/// let display = format!("{}", format_args!("{} foo {:?}", 1, 2));
/// assert_eq!("1 foo 2", display);
/// assert_eq!(display, debug);
/// ```
///
/// [`format()`]: ../../std/fmt/fn.format.html
#[lang = "format_arguments"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone)]
pub struct Arguments<'a> {
    // Format string pieces to print.
    pieces: &'a [&'static str],

    // Placeholder specs, or `None` if all specs are default (as in "{}{}").
    fmt: Option<&'a [rt::Placeholder]>,

    // Dynamic arguments for interpolation, to be interleaved with string
    // pieces. (Every argument is preceded by a string piece.)
    args: &'a [rt::Argument<'a>],
}

/// Used by the format_args!() macro to create a fmt::Arguments object.
#[doc(hidden)]
#[unstable(feature = "fmt_internals", issue = "none")]
impl<'a> Arguments<'a> {
    #[inline]
    pub const fn new_const<const N: usize>(pieces: &'a [&'static str; N]) -> Self {
        const { assert!(N <= 1) };
        Arguments { pieces, fmt: None, args: &[] }
    }

    /// When using the format_args!() macro, this function is used to generate the
    /// Arguments structure.
    #[inline]
    pub const fn new_v1<const P: usize, const A: usize>(
        pieces: &'a [&'static str; P],
        args: &'a [rt::Argument<'a>; A],
    ) -> Arguments<'a> {
        const { assert!(P >= A && P <= A + 1, "invalid args") }
        Arguments { pieces, fmt: None, args }
    }

    /// Specifies nonstandard formatting parameters.
    ///
    /// An `rt::UnsafeArg` is required because the following invariants must be held
    /// in order for this function to be safe:
    /// 1. The `pieces` slice must be at least as long as `fmt`.
    /// 2. Every `rt::Placeholder::position` value within `fmt` must be a valid index of `args`.
    /// 3. Every `rt::Count::Param` within `fmt` must contain a valid index of `args`.
    #[inline]
    pub const fn new_v1_formatted(
        pieces: &'a [&'static str],
        args: &'a [rt::Argument<'a>],
        fmt: &'a [rt::Placeholder],
        _unsafe_arg: rt::UnsafeArg,
    ) -> Arguments<'a> {
        Arguments { pieces, fmt: Some(fmt), args }
    }

    /// Estimates the length of the formatted text.
    ///
    /// This is intended to be used for setting initial `String` capacity
    /// when using `format!`. Note: this is neither the lower nor upper bound.
    #[inline]
    pub fn estimated_capacity(&self) -> usize {
        let pieces_length: usize = self.pieces.iter().map(|x| x.len()).sum();

        if self.args.is_empty() {
            pieces_length
        } else if !self.pieces.is_empty() && self.pieces[0].is_empty() && pieces_length < 16 {
            // If the format string starts with an argument,
            // don't preallocate anything, unless length
            // of pieces is significant.
            0
        } else {
            // There are some arguments, so any additional push
            // will reallocate the string. To avoid that,
            // we're "pre-doubling" the capacity here.
            pieces_length.checked_mul(2).unwrap_or(0)
        }
    }
}

impl<'a> Arguments<'a> {
    /// Gets the formatted string, if it has no arguments to be formatted at runtime.
    ///
    /// This can be used to avoid allocations in some cases.
    ///
    /// # Guarantees
    ///
    /// For `format_args!("just a literal")`, this function is guaranteed to
    /// return `Some("just a literal")`.
    ///
    /// For most cases with placeholders, this function will return `None`.
    ///
    /// However, the compiler may perform optimizations that can cause this
    /// function to return `Some(_)` even if the format string contains
    /// placeholders. For example, `format_args!("Hello, {}!", "world")` may be
    /// optimized to `format_args!("Hello, world!")`, such that `as_str()`
    /// returns `Some("Hello, world!")`.
    ///
    /// The behavior for anything but the trivial case (without placeholders)
    /// is not guaranteed, and should not be relied upon for anything other
    /// than optimization.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::fmt::Arguments;
    ///
    /// fn write_str(_: &str) { /* ... */ }
    ///
    /// fn write_fmt(args: &Arguments<'_>) {
    ///     if let Some(s) = args.as_str() {
    ///         write_str(s)
    ///     } else {
    ///         write_str(&args.to_string());
    ///     }
    /// }
    /// ```
    ///
    /// ```rust
    /// assert_eq!(format_args!("hello").as_str(), Some("hello"));
    /// assert_eq!(format_args!("").as_str(), Some(""));
    /// assert_eq!(format_args!("{:?}", std::env::current_dir()).as_str(), None);
    /// ```
    #[stable(feature = "fmt_as_str", since = "1.52.0")]
    #[rustc_const_stable(feature = "const_arguments_as_str", since = "1.84.0")]
    #[must_use]
    #[inline]
    pub const fn as_str(&self) -> Option<&'static str> {
        match (self.pieces, self.args) {
            ([], []) => Some(""),
            ([s], []) => Some(s),
            _ => None,
        }
    }

    /// Same as [`Arguments::as_str`], but will only return `Some(s)` if it can be determined at compile time.
    #[must_use]
    #[inline]
    fn as_statically_known_str(&self) -> Option<&'static str> {
        let s = self.as_str();
        if core::intrinsics::is_val_statically_known(s.is_some()) { s } else { None }
    }
}

// Manually implementing these results in better error messages.
#[stable(feature = "rust1", since = "1.0.0")]
impl !Send for Arguments<'_> {}
#[stable(feature = "rust1", since = "1.0.0")]
impl !Sync for Arguments<'_> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for Arguments<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        Display::fmt(self, fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Display for Arguments<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        write(fmt.buf, *self)
    }
}

/// `?` formatting.
///
/// `Debug` should format the output in a programmer-facing, debugging context.
///
/// Generally speaking, you should just `derive` a `Debug` implementation.
///
/// When used with the alternate format specifier `#?`, the output is pretty-printed.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
///
/// This trait can be used with `#[derive]` if all fields implement `Debug`. When
/// `derive`d for structs, it will use the name of the `struct`, then `{`, then a
/// comma-separated list of each field's name and `Debug` value, then `}`. For
/// `enum`s, it will use the name of the variant and, if applicable, `(`, then the
/// `Debug` values of the fields, then `)`.
///
/// # Stability
///
/// Derived `Debug` formats are not stable, and so may change with future Rust
/// versions. Additionally, `Debug` implementations of types provided by the
/// standard library (`std`, `core`, `alloc`, etc.) are not stable, and
/// may also change with future Rust versions.
///
/// # Examples
///
/// Deriving an implementation:
///
/// ```
/// #[derive(Debug)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// let origin = Point { x: 0, y: 0 };
///
/// assert_eq!(
///     format!("The origin is: {origin:?}"),
///     "The origin is: Point { x: 0, y: 0 }",
/// );
/// ```
///
/// Manually implementing:
///
/// ```
/// use std::fmt;
///
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl fmt::Debug for Point {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         f.debug_struct("Point")
///          .field("x", &self.x)
///          .field("y", &self.y)
///          .finish()
///     }
/// }
///
/// let origin = Point { x: 0, y: 0 };
///
/// assert_eq!(
///     format!("The origin is: {origin:?}"),
///     "The origin is: Point { x: 0, y: 0 }",
/// );
/// ```
///
/// There are a number of helper methods on the [`Formatter`] struct to help you with manual
/// implementations, such as [`debug_struct`].
///
/// [`debug_struct`]: Formatter::debug_struct
///
/// Types that do not wish to use the standard suite of debug representations
/// provided by the `Formatter` trait (`debug_struct`, `debug_tuple`,
/// `debug_list`, `debug_set`, `debug_map`) can do something totally custom by
/// manually writing an arbitrary representation to the `Formatter`.
///
/// ```
/// # use std::fmt;
/// # struct Point {
/// #     x: i32,
/// #     y: i32,
/// # }
/// #
/// impl fmt::Debug for Point {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "Point [{} {}]", self.x, self.y)
///     }
/// }
/// ```
///
/// `Debug` implementations using either `derive` or the debug builder API
/// on [`Formatter`] support pretty-printing using the alternate flag: `{:#?}`.
///
/// Pretty-printing with `#?`:
///
/// ```
/// #[derive(Debug)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// let origin = Point { x: 0, y: 0 };
///
/// let expected = "The origin is: Point {
///     x: 0,
///     y: 0,
/// }";
/// assert_eq!(format!("The origin is: {origin:#?}"), expected);
/// ```

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(
    on(
        crate_local,
        label = "`{Self}` cannot be formatted using `{{:?}}`",
        note = "add `#[derive(Debug)]` to `{Self}` or manually `impl {Debug} for {Self}`"
    ),
    message = "`{Self}` doesn't implement `{Debug}`",
    label = "`{Self}` cannot be formatted using `{{:?}}` because it doesn't implement `{Debug}`"
)]
#[doc(alias = "{:?}")]
#[rustc_diagnostic_item = "Debug"]
#[rustc_trivial_field_reads]
pub trait Debug {
    #[doc = include_str!("fmt_trait_method_doc.md")]
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Position {
    ///     longitude: f32,
    ///     latitude: f32,
    /// }
    ///
    /// impl fmt::Debug for Position {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         f.debug_tuple("")
    ///          .field(&self.longitude)
    ///          .field(&self.latitude)
    ///          .finish()
    ///     }
    /// }
    ///
    /// let position = Position { longitude: 1.987, latitude: 2.983 };
    /// assert_eq!(format!("{position:?}"), "(1.987, 2.983)");
    ///
    /// assert_eq!(format!("{position:#?}"), "(
    ///     1.987,
    ///     2.983,
    /// )");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

// Separate module to reexport the macro `Debug` from prelude without the trait `Debug`.
pub(crate) mod macros {
    /// Derive macro generating an impl of the trait `Debug`.
    #[rustc_builtin_macro]
    #[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
    #[allow_internal_unstable(core_intrinsics, fmt_helpers_for_derive)]
    pub macro Debug($item:item) {
        /* compiler built-in */
    }
}
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(inline)]
pub use macros::Debug;

/// Format trait for an empty format, `{}`.
///
/// Implementing this trait for a type will automatically implement the
/// [`ToString`][tostring] trait for the type, allowing the usage
/// of the [`.to_string()`][tostring_function] method. Prefer implementing
/// the `Display` trait for a type, rather than [`ToString`][tostring].
///
/// `Display` is similar to [`Debug`], but `Display` is for user-facing
/// output, and so cannot be derived.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
/// [tostring]: ../../std/string/trait.ToString.html
/// [tostring_function]: ../../std/string/trait.ToString.html#tymethod.to_string
///
/// # Internationalization
///
/// Because a type can only have one `Display` implementation, it is often preferable
/// to only implement `Display` when there is a single most "obvious" way that
/// values can be formatted as text. This could mean formatting according to the
/// "invariant" culture and "undefined" locale, or it could mean that the type
/// display is designed for a specific culture/locale, such as developer logs.
///
/// If not all values have a justifiably canonical textual format or if you want
/// to support alternative formats not covered by the standard set of possible
/// [formatting traits], the most flexible approach is display adapters: methods
/// like [`str::escape_default`] or [`Path::display`] which create a wrapper
/// implementing `Display` to output the specific display format.
///
/// [formatting traits]: ../../std/fmt/index.html#formatting-traits
/// [`Path::display`]: ../../std/path/struct.Path.html#method.display
///
/// # Examples
///
/// Implementing `Display` on a type:
///
/// ```
/// use std::fmt;
///
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl fmt::Display for Point {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "({}, {})", self.x, self.y)
///     }
/// }
///
/// let origin = Point { x: 0, y: 0 };
///
/// assert_eq!(format!("The origin is: {origin}"), "The origin is: (0, 0)");
/// ```
#[rustc_on_unimplemented(
    on(
        any(_Self = "std::path::Path", _Self = "std::path::PathBuf"),
        label = "`{Self}` cannot be formatted with the default formatter; call `.display()` on it",
        note = "call `.display()` or `.to_string_lossy()` to safely print paths, \
                as they may contain non-Unicode data"
    ),
    message = "`{Self}` doesn't implement `{Display}`",
    label = "`{Self}` cannot be formatted with the default formatter",
    note = "in format strings you may be able to use `{{:?}}` (or {{:#?}} for pretty-print) instead"
)]
#[doc(alias = "{}")]
#[rustc_diagnostic_item = "Display"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Display {
    #[doc = include_str!("fmt_trait_method_doc.md")]
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Position {
    ///     longitude: f32,
    ///     latitude: f32,
    /// }
    ///
    /// impl fmt::Display for Position {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "({}, {})", self.longitude, self.latitude)
    ///     }
    /// }
    ///
    /// assert_eq!(
    ///     "(1.987, 2.983)",
    ///     format!("{}", Position { longitude: 1.987, latitude: 2.983, }),
    /// );
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// `o` formatting.
///
/// The `Octal` trait should format its output as a number in base-8.
///
/// For primitive signed integers (`i8` to `i128`, and `isize`),
/// negative values are formatted as the two’s complement representation.
///
/// The alternate flag, `#`, adds a `0o` in front of the output.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
///
/// # Examples
///
/// Basic usage with `i32`:
///
/// ```
/// let x = 42; // 42 is '52' in octal
///
/// assert_eq!(format!("{x:o}"), "52");
/// assert_eq!(format!("{x:#o}"), "0o52");
///
/// assert_eq!(format!("{:o}", -16), "37777777760");
/// ```
///
/// Implementing `Octal` on a type:
///
/// ```
/// use std::fmt;
///
/// struct Length(i32);
///
/// impl fmt::Octal for Length {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         let val = self.0;
///
///         fmt::Octal::fmt(&val, f) // delegate to i32's implementation
///     }
/// }
///
/// let l = Length(9);
///
/// assert_eq!(format!("l as octal is: {l:o}"), "l as octal is: 11");
///
/// assert_eq!(format!("l as octal is: {l:#06o}"), "l as octal is: 0o0011");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Octal {
    #[doc = include_str!("fmt_trait_method_doc.md")]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// `b` formatting.
///
/// The `Binary` trait should format its output as a number in binary.
///
/// For primitive signed integers ([`i8`] to [`i128`], and [`isize`]),
/// negative values are formatted as the two’s complement representation.
///
/// The alternate flag, `#`, adds a `0b` in front of the output.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
///
/// # Examples
///
/// Basic usage with [`i32`]:
///
/// ```
/// let x = 42; // 42 is '101010' in binary
///
/// assert_eq!(format!("{x:b}"), "101010");
/// assert_eq!(format!("{x:#b}"), "0b101010");
///
/// assert_eq!(format!("{:b}", -16), "11111111111111111111111111110000");
/// ```
///
/// Implementing `Binary` on a type:
///
/// ```
/// use std::fmt;
///
/// struct Length(i32);
///
/// impl fmt::Binary for Length {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         let val = self.0;
///
///         fmt::Binary::fmt(&val, f) // delegate to i32's implementation
///     }
/// }
///
/// let l = Length(107);
///
/// assert_eq!(format!("l as binary is: {l:b}"), "l as binary is: 1101011");
///
/// assert_eq!(
///     // Note that the `0b` prefix added by `#` is included in the total width, so we
///     // need to add two to correctly display all 32 bits.
///     format!("l as binary is: {l:#034b}"),
///     "l as binary is: 0b00000000000000000000000001101011"
/// );
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Binary {
    #[doc = include_str!("fmt_trait_method_doc.md")]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// `x` formatting.
///
/// The `LowerHex` trait should format its output as a number in hexadecimal, with `a` through `f`
/// in lower case.
///
/// For primitive signed integers (`i8` to `i128`, and `isize`),
/// negative values are formatted as the two’s complement representation.
///
/// The alternate flag, `#`, adds a `0x` in front of the output.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
///
/// # Examples
///
/// Basic usage with `i32`:
///
/// ```
/// let y = 42; // 42 is '2a' in hex
///
/// assert_eq!(format!("{y:x}"), "2a");
/// assert_eq!(format!("{y:#x}"), "0x2a");
///
/// assert_eq!(format!("{:x}", -16), "fffffff0");
/// ```
///
/// Implementing `LowerHex` on a type:
///
/// ```
/// use std::fmt;
///
/// struct Length(i32);
///
/// impl fmt::LowerHex for Length {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         let val = self.0;
///
///         fmt::LowerHex::fmt(&val, f) // delegate to i32's implementation
///     }
/// }
///
/// let l = Length(9);
///
/// assert_eq!(format!("l as hex is: {l:x}"), "l as hex is: 9");
///
/// assert_eq!(format!("l as hex is: {l:#010x}"), "l as hex is: 0x00000009");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait LowerHex {
    #[doc = include_str!("fmt_trait_method_doc.md")]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// `X` formatting.
///
/// The `UpperHex` trait should format its output as a number in hexadecimal, with `A` through `F`
/// in upper case.
///
/// For primitive signed integers (`i8` to `i128`, and `isize`),
/// negative values are formatted as the two’s complement representation.
///
/// The alternate flag, `#`, adds a `0x` in front of the output.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
///
/// # Examples
///
/// Basic usage with `i32`:
///
/// ```
/// let y = 42; // 42 is '2A' in hex
///
/// assert_eq!(format!("{y:X}"), "2A");
/// assert_eq!(format!("{y:#X}"), "0x2A");
///
/// assert_eq!(format!("{:X}", -16), "FFFFFFF0");
/// ```
///
/// Implementing `UpperHex` on a type:
///
/// ```
/// use std::fmt;
///
/// struct Length(i32);
///
/// impl fmt::UpperHex for Length {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         let val = self.0;
///
///         fmt::UpperHex::fmt(&val, f) // delegate to i32's implementation
///     }
/// }
///
/// let l = Length(i32::MAX);
///
/// assert_eq!(format!("l as hex is: {l:X}"), "l as hex is: 7FFFFFFF");
///
/// assert_eq!(format!("l as hex is: {l:#010X}"), "l as hex is: 0x7FFFFFFF");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait UpperHex {
    #[doc = include_str!("fmt_trait_method_doc.md")]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// `p` formatting.
///
/// The `Pointer` trait should format its output as a memory location. This is commonly presented
/// as hexadecimal. For more information on formatters, see [the module-level documentation][module].
///
/// Printing of pointers is not a reliable way to discover how Rust programs are implemented.
/// The act of reading an address changes the program itself, and may change how the data is represented
/// in memory, and may affect which optimizations are applied to the code.
///
/// The printed pointer values are not guaranteed to be stable nor unique identifiers of objects.
/// Rust allows moving values to different memory locations, and may reuse the same memory locations
/// for different purposes.
///
/// There is no guarantee that the printed value can be converted back to a pointer.
///
/// [module]: ../../std/fmt/index.html
///
/// # Examples
///
/// Basic usage with `&i32`:
///
/// ```
/// let x = &42;
///
/// let address = format!("{x:p}"); // this produces something like '0x7f06092ac6d0'
/// ```
///
/// Implementing `Pointer` on a type:
///
/// ```
/// use std::fmt;
///
/// struct Length(i32);
///
/// impl fmt::Pointer for Length {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         // use `as` to convert to a `*const T`, which implements Pointer, which we can use
///
///         let ptr = self as *const Self;
///         fmt::Pointer::fmt(&ptr, f)
///     }
/// }
///
/// let l = Length(42);
///
/// println!("l is in memory here: {l:p}");
///
/// let l_ptr = format!("{l:018p}");
/// assert_eq!(l_ptr.len(), 18);
/// assert_eq!(&l_ptr[..2], "0x");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Pointer"]
pub trait Pointer {
    #[doc = include_str!("fmt_trait_method_doc.md")]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// `e` formatting.
///
/// The `LowerExp` trait should format its output in scientific notation with a lower-case `e`.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
///
/// # Examples
///
/// Basic usage with `f64`:
///
/// ```
/// let x = 42.0; // 42.0 is '4.2e1' in scientific notation
///
/// assert_eq!(format!("{x:e}"), "4.2e1");
/// ```
///
/// Implementing `LowerExp` on a type:
///
/// ```
/// use std::fmt;
///
/// struct Length(i32);
///
/// impl fmt::LowerExp for Length {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         let val = f64::from(self.0);
///         fmt::LowerExp::fmt(&val, f) // delegate to f64's implementation
///     }
/// }
///
/// let l = Length(100);
///
/// assert_eq!(
///     format!("l in scientific notation is: {l:e}"),
///     "l in scientific notation is: 1e2"
/// );
///
/// assert_eq!(
///     format!("l in scientific notation is: {l:05e}"),
///     "l in scientific notation is: 001e2"
/// );
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait LowerExp {
    #[doc = include_str!("fmt_trait_method_doc.md")]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// `E` formatting.
///
/// The `UpperExp` trait should format its output in scientific notation with an upper-case `E`.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
///
/// # Examples
///
/// Basic usage with `f64`:
///
/// ```
/// let x = 42.0; // 42.0 is '4.2E1' in scientific notation
///
/// assert_eq!(format!("{x:E}"), "4.2E1");
/// ```
///
/// Implementing `UpperExp` on a type:
///
/// ```
/// use std::fmt;
///
/// struct Length(i32);
///
/// impl fmt::UpperExp for Length {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         let val = f64::from(self.0);
///         fmt::UpperExp::fmt(&val, f) // delegate to f64's implementation
///     }
/// }
///
/// let l = Length(100);
///
/// assert_eq!(
///     format!("l in scientific notation is: {l:E}"),
///     "l in scientific notation is: 1E2"
/// );
///
/// assert_eq!(
///     format!("l in scientific notation is: {l:05E}"),
///     "l in scientific notation is: 001E2"
/// );
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait UpperExp {
    #[doc = include_str!("fmt_trait_method_doc.md")]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// Takes an output stream and an `Arguments` struct that can be precompiled with
/// the `format_args!` macro.
///
/// The arguments will be formatted according to the specified format string
/// into the output stream provided.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::fmt;
///
/// let mut output = String::new();
/// fmt::write(&mut output, format_args!("Hello {}!", "world"))
///     .expect("Error occurred while trying to write in String");
/// assert_eq!(output, "Hello world!");
/// ```
///
/// Please note that using [`write!`] might be preferable. Example:
///
/// ```
/// use std::fmt::Write;
///
/// let mut output = String::new();
/// write!(&mut output, "Hello {}!", "world")
///     .expect("Error occurred while trying to write in String");
/// assert_eq!(output, "Hello world!");
/// ```
///
/// [`write!`]: crate::write!
#[stable(feature = "rust1", since = "1.0.0")]
pub fn write(output: &mut dyn Write, args: Arguments<'_>) -> Result {
    let mut formatter = Formatter::new(output, FormattingOptions::new());
    let mut idx = 0;

    match args.fmt {
        None => {
            // We can use default formatting parameters for all arguments.
            for (i, arg) in args.args.iter().enumerate() {
                // SAFETY: args.args and args.pieces come from the same Arguments,
                // which guarantees the indexes are always within bounds.
                let piece = unsafe { args.pieces.get_unchecked(i) };
                if !piece.is_empty() {
                    formatter.buf.write_str(*piece)?;
                }

                // SAFETY: There are no formatting parameters and hence no
                // count arguments.
                unsafe {
                    arg.fmt(&mut formatter)?;
                }
                idx += 1;
            }
        }
        Some(fmt) => {
            // Every spec has a corresponding argument that is preceded by
            // a string piece.
            for (i, arg) in fmt.iter().enumerate() {
                // SAFETY: fmt and args.pieces come from the same Arguments,
                // which guarantees the indexes are always within bounds.
                let piece = unsafe { args.pieces.get_unchecked(i) };
                if !piece.is_empty() {
                    formatter.buf.write_str(*piece)?;
                }
                // SAFETY: arg and args.args come from the same Arguments,
                // which guarantees the indexes are always within bounds.
                unsafe { run(&mut formatter, arg, args.args) }?;
                idx += 1;
            }
        }
    }

    // There can be only one trailing string piece left.
    if let Some(piece) = args.pieces.get(idx) {
        formatter.buf.write_str(*piece)?;
    }

    Ok(())
}

unsafe fn run(fmt: &mut Formatter<'_>, arg: &rt::Placeholder, args: &[rt::Argument<'_>]) -> Result {
    fmt.options.fill = arg.fill;
    fmt.options.align = arg.align.into();
    fmt.options.flags = arg.flags;
    // SAFETY: arg and args come from the same Arguments,
    // which guarantees the indexes are always within bounds.
    unsafe {
        fmt.options.width = getcount(args, &arg.width);
        fmt.options.precision = getcount(args, &arg.precision);
    }

    // Extract the correct argument
    debug_assert!(arg.position < args.len());
    // SAFETY: arg and args come from the same Arguments,
    // which guarantees its index is always within bounds.
    let value = unsafe { args.get_unchecked(arg.position) };

    // Then actually do some printing
    // SAFETY: this is a placeholder argument.
    unsafe { value.fmt(fmt) }
}

unsafe fn getcount(args: &[rt::Argument<'_>], cnt: &rt::Count) -> Option<usize> {
    match *cnt {
        rt::Count::Is(n) => Some(n),
        rt::Count::Implied => None,
        rt::Count::Param(i) => {
            debug_assert!(i < args.len());
            // SAFETY: cnt and args come from the same Arguments,
            // which guarantees this index is always within bounds.
            unsafe { args.get_unchecked(i).as_usize() }
        }
    }
}

/// Padding after the end of something. Returned by `Formatter::padding`.
#[must_use = "don't forget to write the post padding"]
pub(crate) struct PostPadding {
    fill: char,
    padding: usize,
}

impl PostPadding {
    fn new(fill: char, padding: usize) -> PostPadding {
        PostPadding { fill, padding }
    }

    /// Writes this post padding.
    pub(crate) fn write(self, f: &mut Formatter<'_>) -> Result {
        for _ in 0..self.padding {
            f.buf.write_char(self.fill)?;
        }
        Ok(())
    }
}

impl<'a> Formatter<'a> {
    fn wrap_buf<'b, 'c, F>(&'b mut self, wrap: F) -> Formatter<'c>
    where
        'b: 'c,
        F: FnOnce(&'b mut (dyn Write + 'b)) -> &'c mut (dyn Write + 'c),
    {
        Formatter {
            // We want to change this
            buf: wrap(self.buf),

            // And preserve these
            options: self.options,
        }
    }

    // Helper methods used for padding and processing formatting arguments that
    // all formatting traits can use.

    /// Performs the correct padding for an integer which has already been
    /// emitted into a str. The str should *not* contain the sign for the
    /// integer, that will be added by this method.
    ///
    /// # Arguments
    ///
    /// * is_nonnegative - whether the original integer was either positive or zero.
    /// * prefix - if the '#' character (Alternate) is provided, this
    ///   is the prefix to put in front of the number.
    /// * buf - the byte array that the number has been formatted into
    ///
    /// This function will correctly account for the flags provided as well as
    /// the minimum width. It will not take precision into account.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo { nb: i32 }
    ///
    /// impl Foo {
    ///     fn new(nb: i32) -> Foo {
    ///         Foo {
    ///             nb,
    ///         }
    ///     }
    /// }
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         // We need to remove "-" from the number output.
    ///         let tmp = self.nb.abs().to_string();
    ///
    ///         formatter.pad_integral(self.nb >= 0, "Foo ", &tmp)
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{}", Foo::new(2)), "2");
    /// assert_eq!(format!("{}", Foo::new(-1)), "-1");
    /// assert_eq!(format!("{}", Foo::new(0)), "0");
    /// assert_eq!(format!("{:#}", Foo::new(-1)), "-Foo 1");
    /// assert_eq!(format!("{:0>#8}", Foo::new(-1)), "00-Foo 1");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pad_integral(&mut self, is_nonnegative: bool, prefix: &str, buf: &str) -> Result {
        let mut width = buf.len();

        let mut sign = None;
        if !is_nonnegative {
            sign = Some('-');
            width += 1;
        } else if self.sign_plus() {
            sign = Some('+');
            width += 1;
        }

        let prefix = if self.alternate() {
            width += prefix.chars().count();
            Some(prefix)
        } else {
            None
        };

        // Writes the sign if it exists, and then the prefix if it was requested
        #[inline(never)]
        fn write_prefix(f: &mut Formatter<'_>, sign: Option<char>, prefix: Option<&str>) -> Result {
            if let Some(c) = sign {
                f.buf.write_char(c)?;
            }
            if let Some(prefix) = prefix { f.buf.write_str(prefix) } else { Ok(()) }
        }

        // The `width` field is more of a `min-width` parameter at this point.
        match self.options.width {
            // If there's no minimum length requirements then we can just
            // write the bytes.
            None => {
                write_prefix(self, sign, prefix)?;
                self.buf.write_str(buf)
            }
            // Check if we're over the minimum width, if so then we can also
            // just write the bytes.
            Some(min) if width >= min => {
                write_prefix(self, sign, prefix)?;
                self.buf.write_str(buf)
            }
            // The sign and prefix goes before the padding if the fill character
            // is zero
            Some(min) if self.sign_aware_zero_pad() => {
                let old_fill = crate::mem::replace(&mut self.options.fill, '0');
                let old_align =
                    crate::mem::replace(&mut self.options.align, Some(Alignment::Right));
                write_prefix(self, sign, prefix)?;
                let post_padding = self.padding(min - width, Alignment::Right)?;
                self.buf.write_str(buf)?;
                post_padding.write(self)?;
                self.options.fill = old_fill;
                self.options.align = old_align;
                Ok(())
            }
            // Otherwise, the sign and prefix goes after the padding
            Some(min) => {
                let post_padding = self.padding(min - width, Alignment::Right)?;
                write_prefix(self, sign, prefix)?;
                self.buf.write_str(buf)?;
                post_padding.write(self)
            }
        }
    }

    /// Takes a string slice and emits it to the internal buffer after applying
    /// the relevant formatting flags specified.
    ///
    /// The flags recognized for generic strings are:
    ///
    /// * width - the minimum width of what to emit
    /// * fill/align - what to emit and where to emit it if the string
    ///                provided needs to be padded
    /// * precision - the maximum length to emit, the string is truncated if it
    ///               is longer than this length
    ///
    /// Notably this function ignores the `flag` parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo;
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         formatter.pad("Foo")
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{Foo:<4}"), "Foo ");
    /// assert_eq!(format!("{Foo:0>4}"), "0Foo");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pad(&mut self, s: &str) -> Result {
        // Make sure there's a fast path up front
        if self.options.width.is_none() && self.options.precision.is_none() {
            return self.buf.write_str(s);
        }
        // The `precision` field can be interpreted as a `max-width` for the
        // string being formatted.
        let s = if let Some(max) = self.options.precision {
            // If our string is longer that the precision, then we must have
            // truncation. However other flags like `fill`, `width` and `align`
            // must act as always.
            if let Some((i, _)) = s.char_indices().nth(max) {
                // LLVM here can't prove that `..i` won't panic `&s[..i]`, but
                // we know that it can't panic. Use `get` + `unwrap_or` to avoid
                // `unsafe` and otherwise don't emit any panic-related code
                // here.
                s.get(..i).unwrap_or(s)
            } else {
                &s
            }
        } else {
            &s
        };
        // The `width` field is more of a `min-width` parameter at this point.
        match self.options.width {
            // If we're under the maximum length, and there's no minimum length
            // requirements, then we can just emit the string
            None => self.buf.write_str(s),
            Some(width) => {
                let chars_count = s.chars().count();
                // If we're under the maximum width, check if we're over the minimum
                // width, if so it's as easy as just emitting the string.
                if chars_count >= width {
                    self.buf.write_str(s)
                }
                // If we're under both the maximum and the minimum width, then fill
                // up the minimum width with the specified string + some alignment.
                else {
                    let align = Alignment::Left;
                    let post_padding = self.padding(width - chars_count, align)?;
                    self.buf.write_str(s)?;
                    post_padding.write(self)
                }
            }
        }
    }

    /// Writes the pre-padding and returns the unwritten post-padding.
    ///
    /// Callers are responsible for ensuring post-padding is written after the
    /// thing that is being padded.
    pub(crate) fn padding(
        &mut self,
        padding: usize,
        default: Alignment,
    ) -> result::Result<PostPadding, Error> {
        let align = self.align().unwrap_or(default);

        let (pre_pad, post_pad) = match align {
            Alignment::Left => (0, padding),
            Alignment::Right => (padding, 0),
            Alignment::Center => (padding / 2, (padding + 1) / 2),
        };

        for _ in 0..pre_pad {
            self.buf.write_char(self.options.fill)?;
        }

        Ok(PostPadding::new(self.options.fill, post_pad))
    }

    /// Takes the formatted parts and applies the padding.
    ///
    /// Assumes that the caller already has rendered the parts with required precision,
    /// so that `self.precision` can be ignored.
    ///
    /// # Safety
    ///
    /// Any `numfmt::Part::Copy` parts in `formatted` must contain valid UTF-8.
    unsafe fn pad_formatted_parts(&mut self, formatted: &numfmt::Formatted<'_>) -> Result {
        if let Some(mut width) = self.options.width {
            // for the sign-aware zero padding, we render the sign first and
            // behave as if we had no sign from the beginning.
            let mut formatted = formatted.clone();
            let old_fill = self.options.fill;
            let old_align = self.options.align;
            if self.sign_aware_zero_pad() {
                // a sign always goes first
                let sign = formatted.sign;
                self.buf.write_str(sign)?;

                // remove the sign from the formatted parts
                formatted.sign = "";
                width = width.saturating_sub(sign.len());
                self.options.fill = '0';
                self.options.align = Some(Alignment::Right);
            }

            // remaining parts go through the ordinary padding process.
            let len = formatted.len();
            let ret = if width <= len {
                // no padding
                // SAFETY: Per the precondition.
                unsafe { self.write_formatted_parts(&formatted) }
            } else {
                let post_padding = self.padding(width - len, Alignment::Right)?;
                // SAFETY: Per the precondition.
                unsafe {
                    self.write_formatted_parts(&formatted)?;
                }
                post_padding.write(self)
            };
            self.options.fill = old_fill;
            self.options.align = old_align;
            ret
        } else {
            // this is the common case and we take a shortcut
            // SAFETY: Per the precondition.
            unsafe { self.write_formatted_parts(formatted) }
        }
    }

    /// # Safety
    ///
    /// Any `numfmt::Part::Copy` parts in `formatted` must contain valid UTF-8.
    unsafe fn write_formatted_parts(&mut self, formatted: &numfmt::Formatted<'_>) -> Result {
        unsafe fn write_bytes(buf: &mut dyn Write, s: &[u8]) -> Result {
            // SAFETY: This is used for `numfmt::Part::Num` and `numfmt::Part::Copy`.
            // It's safe to use for `numfmt::Part::Num` since every char `c` is between
            // `b'0'` and `b'9'`, which means `s` is valid UTF-8. It's safe to use for
            // `numfmt::Part::Copy` due to this function's precondition.
            buf.write_str(unsafe { str::from_utf8_unchecked(s) })
        }

        if !formatted.sign.is_empty() {
            self.buf.write_str(formatted.sign)?;
        }
        for part in formatted.parts {
            match *part {
                numfmt::Part::Zero(mut nzeroes) => {
                    const ZEROES: &str = // 64 zeroes
                        "0000000000000000000000000000000000000000000000000000000000000000";
                    while nzeroes > ZEROES.len() {
                        self.buf.write_str(ZEROES)?;
                        nzeroes -= ZEROES.len();
                    }
                    if nzeroes > 0 {
                        self.buf.write_str(&ZEROES[..nzeroes])?;
                    }
                }
                numfmt::Part::Num(mut v) => {
                    let mut s = [0; 5];
                    let len = part.len();
                    for c in s[..len].iter_mut().rev() {
                        *c = b'0' + (v % 10) as u8;
                        v /= 10;
                    }
                    // SAFETY: Per the precondition.
                    unsafe {
                        write_bytes(self.buf, &s[..len])?;
                    }
                }
                // SAFETY: Per the precondition.
                numfmt::Part::Copy(buf) => unsafe {
                    write_bytes(self.buf, buf)?;
                },
            }
        }
        Ok(())
    }

    /// Writes some data to the underlying buffer contained within this
    /// formatter.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo;
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         formatter.write_str("Foo")
    ///         // This is equivalent to:
    ///         // write!(formatter, "Foo")
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{Foo}"), "Foo");
    /// assert_eq!(format!("{Foo:0>8}"), "Foo");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn write_str(&mut self, data: &str) -> Result {
        self.buf.write_str(data)
    }

    /// Glue for usage of the [`write!`] macro with implementors of this trait.
    ///
    /// This method should generally not be invoked manually, but rather through
    /// the [`write!`] macro itself.
    ///
    /// Writes some formatted information into this instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo(i32);
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         formatter.write_fmt(format_args!("Foo {}", self.0))
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{}", Foo(-1)), "Foo -1");
    /// assert_eq!(format!("{:0>8}", Foo(2)), "Foo 2");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn write_fmt(&mut self, fmt: Arguments<'_>) -> Result {
        if let Some(s) = fmt.as_statically_known_str() {
            self.buf.write_str(s)
        } else {
            write(self.buf, fmt)
        }
    }

    /// Returns flags for formatting.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(
        since = "1.24.0",
        note = "use the `sign_plus`, `sign_minus`, `alternate`, \
                or `sign_aware_zero_pad` methods instead"
    )]
    pub fn flags(&self) -> u32 {
        self.options.flags
    }

    /// Returns the character used as 'fill' whenever there is alignment.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo;
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         let c = formatter.fill();
    ///         if let Some(width) = formatter.width() {
    ///             for _ in 0..width {
    ///                 write!(formatter, "{c}")?;
    ///             }
    ///             Ok(())
    ///         } else {
    ///             write!(formatter, "{c}")
    ///         }
    ///     }
    /// }
    ///
    /// // We set alignment to the right with ">".
    /// assert_eq!(format!("{Foo:G>3}"), "GGG");
    /// assert_eq!(format!("{Foo:t>6}"), "tttttt");
    /// ```
    #[must_use]
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn fill(&self) -> char {
        self.options.fill
    }

    /// Returns a flag indicating what form of alignment was requested.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::{self, Alignment};
    ///
    /// struct Foo;
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         let s = if let Some(s) = formatter.align() {
    ///             match s {
    ///                 Alignment::Left    => "left",
    ///                 Alignment::Right   => "right",
    ///                 Alignment::Center  => "center",
    ///             }
    ///         } else {
    ///             "into the void"
    ///         };
    ///         write!(formatter, "{s}")
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{Foo:<}"), "left");
    /// assert_eq!(format!("{Foo:>}"), "right");
    /// assert_eq!(format!("{Foo:^}"), "center");
    /// assert_eq!(format!("{Foo}"), "into the void");
    /// ```
    #[must_use]
    #[stable(feature = "fmt_flags_align", since = "1.28.0")]
    pub fn align(&self) -> Option<Alignment> {
        self.options.align
    }

    /// Returns the optionally specified integer width that the output should be.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo(i32);
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         if let Some(width) = formatter.width() {
    ///             // If we received a width, we use it
    ///             write!(formatter, "{:width$}", format!("Foo({})", self.0), width = width)
    ///         } else {
    ///             // Otherwise we do nothing special
    ///             write!(formatter, "Foo({})", self.0)
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{:10}", Foo(23)), "Foo(23)   ");
    /// assert_eq!(format!("{}", Foo(23)), "Foo(23)");
    /// ```
    #[must_use]
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn width(&self) -> Option<usize> {
        self.options.width
    }

    /// Returns the optionally specified precision for numeric types.
    /// Alternatively, the maximum width for string types.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo(f32);
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         if let Some(precision) = formatter.precision() {
    ///             // If we received a precision, we use it.
    ///             write!(formatter, "Foo({1:.*})", precision, self.0)
    ///         } else {
    ///             // Otherwise we default to 2.
    ///             write!(formatter, "Foo({:.2})", self.0)
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{:.4}", Foo(23.2)), "Foo(23.2000)");
    /// assert_eq!(format!("{}", Foo(23.2)), "Foo(23.20)");
    /// ```
    #[must_use]
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn precision(&self) -> Option<usize> {
        self.options.precision
    }

    /// Determines if the `+` flag was specified.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo(i32);
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         if formatter.sign_plus() {
    ///             write!(formatter,
    ///                    "Foo({}{})",
    ///                    if self.0 < 0 { '-' } else { '+' },
    ///                    self.0.abs())
    ///         } else {
    ///             write!(formatter, "Foo({})", self.0)
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{:+}", Foo(23)), "Foo(+23)");
    /// assert_eq!(format!("{:+}", Foo(-23)), "Foo(-23)");
    /// assert_eq!(format!("{}", Foo(23)), "Foo(23)");
    /// ```
    #[must_use]
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn sign_plus(&self) -> bool {
        self.options.flags & (1 << rt::Flag::SignPlus as u32) != 0
    }

    /// Determines if the `-` flag was specified.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo(i32);
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         if formatter.sign_minus() {
    ///             // You want a minus sign? Have one!
    ///             write!(formatter, "-Foo({})", self.0)
    ///         } else {
    ///             write!(formatter, "Foo({})", self.0)
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{:-}", Foo(23)), "-Foo(23)");
    /// assert_eq!(format!("{}", Foo(23)), "Foo(23)");
    /// ```
    #[must_use]
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn sign_minus(&self) -> bool {
        self.options.flags & (1 << rt::Flag::SignMinus as u32) != 0
    }

    /// Determines if the `#` flag was specified.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo(i32);
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         if formatter.alternate() {
    ///             write!(formatter, "Foo({})", self.0)
    ///         } else {
    ///             write!(formatter, "{}", self.0)
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{:#}", Foo(23)), "Foo(23)");
    /// assert_eq!(format!("{}", Foo(23)), "23");
    /// ```
    #[must_use]
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn alternate(&self) -> bool {
        self.options.flags & (1 << rt::Flag::Alternate as u32) != 0
    }

    /// Determines if the `0` flag was specified.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo(i32);
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         assert!(formatter.sign_aware_zero_pad());
    ///         assert_eq!(formatter.width(), Some(4));
    ///         // We ignore the formatter's options.
    ///         write!(formatter, "{}", self.0)
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{:04}", Foo(23)), "23");
    /// ```
    #[must_use]
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn sign_aware_zero_pad(&self) -> bool {
        self.options.flags & (1 << rt::Flag::SignAwareZeroPad as u32) != 0
    }

    // FIXME: Decide what public API we want for these two flags.
    // https://github.com/rust-lang/rust/issues/48584
    fn debug_lower_hex(&self) -> bool {
        self.options.flags & (1 << rt::Flag::DebugLowerHex as u32) != 0
    }

    fn debug_upper_hex(&self) -> bool {
        self.options.flags & (1 << rt::Flag::DebugUpperHex as u32) != 0
    }

    /// Creates a [`DebugStruct`] builder designed to assist with creation of
    /// [`fmt::Debug`] implementations for structs.
    ///
    /// [`fmt::Debug`]: self::Debug
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::fmt;
    /// use std::net::Ipv4Addr;
    ///
    /// struct Foo {
    ///     bar: i32,
    ///     baz: String,
    ///     addr: Ipv4Addr,
    /// }
    ///
    /// impl fmt::Debug for Foo {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         fmt.debug_struct("Foo")
    ///             .field("bar", &self.bar)
    ///             .field("baz", &self.baz)
    ///             .field("addr", &format_args!("{}", self.addr))
    ///             .finish()
    ///     }
    /// }
    ///
    /// assert_eq!(
    ///     "Foo { bar: 10, baz: \"Hello World\", addr: 127.0.0.1 }",
    ///     format!("{:?}", Foo {
    ///         bar: 10,
    ///         baz: "Hello World".to_string(),
    ///         addr: Ipv4Addr::new(127, 0, 0, 1),
    ///     })
    /// );
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    pub fn debug_struct<'b>(&'b mut self, name: &str) -> DebugStruct<'b, 'a> {
        builders::debug_struct_new(self, name)
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_struct_fields_finish` is more general, but this is
    /// faster for 1 field.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_struct_field1_finish<'b>(
        &'b mut self,
        name: &str,
        name1: &str,
        value1: &dyn Debug,
    ) -> Result {
        let mut builder = builders::debug_struct_new(self, name);
        builder.field(name1, value1);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_struct_fields_finish` is more general, but this is
    /// faster for 2 fields.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_struct_field2_finish<'b>(
        &'b mut self,
        name: &str,
        name1: &str,
        value1: &dyn Debug,
        name2: &str,
        value2: &dyn Debug,
    ) -> Result {
        let mut builder = builders::debug_struct_new(self, name);
        builder.field(name1, value1);
        builder.field(name2, value2);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_struct_fields_finish` is more general, but this is
    /// faster for 3 fields.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_struct_field3_finish<'b>(
        &'b mut self,
        name: &str,
        name1: &str,
        value1: &dyn Debug,
        name2: &str,
        value2: &dyn Debug,
        name3: &str,
        value3: &dyn Debug,
    ) -> Result {
        let mut builder = builders::debug_struct_new(self, name);
        builder.field(name1, value1);
        builder.field(name2, value2);
        builder.field(name3, value3);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_struct_fields_finish` is more general, but this is
    /// faster for 4 fields.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_struct_field4_finish<'b>(
        &'b mut self,
        name: &str,
        name1: &str,
        value1: &dyn Debug,
        name2: &str,
        value2: &dyn Debug,
        name3: &str,
        value3: &dyn Debug,
        name4: &str,
        value4: &dyn Debug,
    ) -> Result {
        let mut builder = builders::debug_struct_new(self, name);
        builder.field(name1, value1);
        builder.field(name2, value2);
        builder.field(name3, value3);
        builder.field(name4, value4);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_struct_fields_finish` is more general, but this is
    /// faster for 5 fields.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_struct_field5_finish<'b>(
        &'b mut self,
        name: &str,
        name1: &str,
        value1: &dyn Debug,
        name2: &str,
        value2: &dyn Debug,
        name3: &str,
        value3: &dyn Debug,
        name4: &str,
        value4: &dyn Debug,
        name5: &str,
        value5: &dyn Debug,
    ) -> Result {
        let mut builder = builders::debug_struct_new(self, name);
        builder.field(name1, value1);
        builder.field(name2, value2);
        builder.field(name3, value3);
        builder.field(name4, value4);
        builder.field(name5, value5);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller binaries.
    /// For the cases not covered by `debug_struct_field[12345]_finish`.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_struct_fields_finish<'b>(
        &'b mut self,
        name: &str,
        names: &[&str],
        values: &[&dyn Debug],
    ) -> Result {
        assert_eq!(names.len(), values.len());
        let mut builder = builders::debug_struct_new(self, name);
        for (name, value) in iter::zip(names, values) {
            builder.field(name, value);
        }
        builder.finish()
    }

    /// Creates a `DebugTuple` builder designed to assist with creation of
    /// `fmt::Debug` implementations for tuple structs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::fmt;
    /// use std::marker::PhantomData;
    ///
    /// struct Foo<T>(i32, String, PhantomData<T>);
    ///
    /// impl<T> fmt::Debug for Foo<T> {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         fmt.debug_tuple("Foo")
    ///             .field(&self.0)
    ///             .field(&self.1)
    ///             .field(&format_args!("_"))
    ///             .finish()
    ///     }
    /// }
    ///
    /// assert_eq!(
    ///     "Foo(10, \"Hello\", _)",
    ///     format!("{:?}", Foo(10, "Hello".to_string(), PhantomData::<u8>))
    /// );
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    pub fn debug_tuple<'b>(&'b mut self, name: &str) -> DebugTuple<'b, 'a> {
        builders::debug_tuple_new(self, name)
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_tuple_fields_finish` is more general, but this is faster
    /// for 1 field.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_tuple_field1_finish<'b>(&'b mut self, name: &str, value1: &dyn Debug) -> Result {
        let mut builder = builders::debug_tuple_new(self, name);
        builder.field(value1);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_tuple_fields_finish` is more general, but this is faster
    /// for 2 fields.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_tuple_field2_finish<'b>(
        &'b mut self,
        name: &str,
        value1: &dyn Debug,
        value2: &dyn Debug,
    ) -> Result {
        let mut builder = builders::debug_tuple_new(self, name);
        builder.field(value1);
        builder.field(value2);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_tuple_fields_finish` is more general, but this is faster
    /// for 3 fields.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_tuple_field3_finish<'b>(
        &'b mut self,
        name: &str,
        value1: &dyn Debug,
        value2: &dyn Debug,
        value3: &dyn Debug,
    ) -> Result {
        let mut builder = builders::debug_tuple_new(self, name);
        builder.field(value1);
        builder.field(value2);
        builder.field(value3);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_tuple_fields_finish` is more general, but this is faster
    /// for 4 fields.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_tuple_field4_finish<'b>(
        &'b mut self,
        name: &str,
        value1: &dyn Debug,
        value2: &dyn Debug,
        value3: &dyn Debug,
        value4: &dyn Debug,
    ) -> Result {
        let mut builder = builders::debug_tuple_new(self, name);
        builder.field(value1);
        builder.field(value2);
        builder.field(value3);
        builder.field(value4);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. `debug_tuple_fields_finish` is more general, but this is faster
    /// for 5 fields.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_tuple_field5_finish<'b>(
        &'b mut self,
        name: &str,
        value1: &dyn Debug,
        value2: &dyn Debug,
        value3: &dyn Debug,
        value4: &dyn Debug,
        value5: &dyn Debug,
    ) -> Result {
        let mut builder = builders::debug_tuple_new(self, name);
        builder.field(value1);
        builder.field(value2);
        builder.field(value3);
        builder.field(value4);
        builder.field(value5);
        builder.finish()
    }

    /// Shrinks `derive(Debug)` code, for faster compilation and smaller
    /// binaries. For the cases not covered by `debug_tuple_field[12345]_finish`.
    #[doc(hidden)]
    #[unstable(feature = "fmt_helpers_for_derive", issue = "none")]
    pub fn debug_tuple_fields_finish<'b>(
        &'b mut self,
        name: &str,
        values: &[&dyn Debug],
    ) -> Result {
        let mut builder = builders::debug_tuple_new(self, name);
        for value in values {
            builder.field(value);
        }
        builder.finish()
    }

    /// Creates a `DebugList` builder designed to assist with creation of
    /// `fmt::Debug` implementations for list-like structures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::fmt;
    ///
    /// struct Foo(Vec<i32>);
    ///
    /// impl fmt::Debug for Foo {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         fmt.debug_list().entries(self.0.iter()).finish()
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{:?}", Foo(vec![10, 11])), "[10, 11]");
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    pub fn debug_list<'b>(&'b mut self) -> DebugList<'b, 'a> {
        builders::debug_list_new(self)
    }

    /// Creates a `DebugSet` builder designed to assist with creation of
    /// `fmt::Debug` implementations for set-like structures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::fmt;
    ///
    /// struct Foo(Vec<i32>);
    ///
    /// impl fmt::Debug for Foo {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         fmt.debug_set().entries(self.0.iter()).finish()
    ///     }
    /// }
    ///
    /// assert_eq!(format!("{:?}", Foo(vec![10, 11])), "{10, 11}");
    /// ```
    ///
    /// [`format_args!`]: crate::format_args
    ///
    /// In this more complex example, we use [`format_args!`] and `.debug_set()`
    /// to build a list of match arms:
    ///
    /// ```rust
    /// use std::fmt;
    ///
    /// struct Arm<'a, L, R>(&'a (L, R));
    /// struct Table<'a, K, V>(&'a [(K, V)], V);
    ///
    /// impl<'a, L, R> fmt::Debug for Arm<'a, L, R>
    /// where
    ///     L: 'a + fmt::Debug, R: 'a + fmt::Debug
    /// {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         L::fmt(&(self.0).0, fmt)?;
    ///         fmt.write_str(" => ")?;
    ///         R::fmt(&(self.0).1, fmt)
    ///     }
    /// }
    ///
    /// impl<'a, K, V> fmt::Debug for Table<'a, K, V>
    /// where
    ///     K: 'a + fmt::Debug, V: 'a + fmt::Debug
    /// {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         fmt.debug_set()
    ///         .entries(self.0.iter().map(Arm))
    ///         .entry(&Arm(&(format_args!("_"), &self.1)))
    ///         .finish()
    ///     }
    /// }
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    pub fn debug_set<'b>(&'b mut self) -> DebugSet<'b, 'a> {
        builders::debug_set_new(self)
    }

    /// Creates a `DebugMap` builder designed to assist with creation of
    /// `fmt::Debug` implementations for map-like structures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::fmt;
    ///
    /// struct Foo(Vec<(String, i32)>);
    ///
    /// impl fmt::Debug for Foo {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         fmt.debug_map().entries(self.0.iter().map(|&(ref k, ref v)| (k, v))).finish()
    ///     }
    /// }
    ///
    /// assert_eq!(
    ///     format!("{:?}",  Foo(vec![("A".to_string(), 10), ("B".to_string(), 11)])),
    ///     r#"{"A": 10, "B": 11}"#
    ///  );
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    pub fn debug_map<'b>(&'b mut self) -> DebugMap<'b, 'a> {
        builders::debug_map_new(self)
    }

    /// Returns the sign of this formatter (`+` or `-`).
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn sign(&self) -> Option<Sign> {
        self.options.get_sign()
    }

    /// Returns the formatting options this formatter corresponds to.
    #[unstable(feature = "formatting_options", issue = "118117")]
    pub const fn options(&self) -> FormattingOptions {
        self.options
    }
}

#[stable(since = "1.2.0", feature = "formatter_write")]
impl Write for Formatter<'_> {
    fn write_str(&mut self, s: &str) -> Result {
        self.buf.write_str(s)
    }

    fn write_char(&mut self, c: char) -> Result {
        self.buf.write_char(c)
    }

    #[inline]
    fn write_fmt(&mut self, args: Arguments<'_>) -> Result {
        if let Some(s) = args.as_statically_known_str() {
            self.buf.write_str(s)
        } else {
            write(self.buf, args)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Display::fmt("an error occurred when formatting an argument", f)
    }
}

// Implementations of the core formatting traits

macro_rules! fmt_refs {
    ($($tr:ident),*) => {
        $(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<T: ?Sized + $tr> $tr for &T {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result { $tr::fmt(&**self, f) }
        }
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<T: ?Sized + $tr> $tr for &mut T {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result { $tr::fmt(&**self, f) }
        }
        )*
    }
}

fmt_refs! { Debug, Display, Octal, Binary, LowerHex, UpperHex, LowerExp, UpperExp }

#[unstable(feature = "never_type", issue = "35121")]
impl Debug for ! {
    #[inline]
    fn fmt(&self, _: &mut Formatter<'_>) -> Result {
        *self
    }
}

#[unstable(feature = "never_type", issue = "35121")]
impl Display for ! {
    #[inline]
    fn fmt(&self, _: &mut Formatter<'_>) -> Result {
        *self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for bool {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Display::fmt(self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Display for bool {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Display::fmt(if *self { "true" } else { "false" }, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for str {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.write_char('"')?;

        // substring we know is printable
        let mut printable_range = 0..0;

        fn needs_escape(b: u8) -> bool {
            b > 0x7E || b < 0x20 || b == b'\\' || b == b'"'
        }

        // the loop here first skips over runs of printable ASCII as a fast path.
        // other chars (unicode, or ASCII that needs escaping) are then handled per-`char`.
        let mut rest = self;
        while rest.len() > 0 {
            let Some(non_printable_start) = rest.as_bytes().iter().position(|&b| needs_escape(b))
            else {
                printable_range.end += rest.len();
                break;
            };

            printable_range.end += non_printable_start;
            // SAFETY: the position was derived from an iterator, so is known to be within bounds, and at a char boundary
            rest = unsafe { rest.get_unchecked(non_printable_start..) };

            let mut chars = rest.chars();
            if let Some(c) = chars.next() {
                let esc = c.escape_debug_ext(EscapeDebugExtArgs {
                    escape_grapheme_extended: true,
                    escape_single_quote: false,
                    escape_double_quote: true,
                });
                if esc.len() != 1 {
                    f.write_str(&self[printable_range.clone()])?;
                    Display::fmt(&esc, f)?;
                    printable_range.start = printable_range.end + c.len_utf8();
                }
                printable_range.end += c.len_utf8();
            }
            rest = chars.as_str();
        }

        f.write_str(&self[printable_range])?;

        f.write_char('"')
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Display for str {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.pad(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for char {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.write_char('\'')?;
        let esc = self.escape_debug_ext(EscapeDebugExtArgs {
            escape_grapheme_extended: true,
            escape_single_quote: true,
            escape_double_quote: false,
        });
        Display::fmt(&esc, f)?;
        f.write_char('\'')
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Display for char {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if f.options.width.is_none() && f.options.precision.is_none() {
            f.write_char(*self)
        } else {
            f.pad(self.encode_utf8(&mut [0; MAX_LEN_UTF8]))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Pointer for *const T {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        pointer_fmt_inner(self.expose_provenance(), f)
    }
}

/// Since the formatting will be identical for all pointer types, uses a
/// non-monomorphized implementation for the actual formatting to reduce the
/// amount of codegen work needed.
///
/// This uses `ptr_addr: usize` and not `ptr: *const ()` to be able to use this for
/// `fn(...) -> ...` without using [problematic] "Oxford Casts".
///
/// [problematic]: https://github.com/rust-lang/rust/issues/95489
pub(crate) fn pointer_fmt_inner(ptr_addr: usize, f: &mut Formatter<'_>) -> Result {
    let old_width = f.options.width;
    let old_flags = f.options.flags;

    // The alternate flag is already treated by LowerHex as being special-
    // it denotes whether to prefix with 0x. We use it to work out whether
    // or not to zero extend, and then unconditionally set it to get the
    // prefix.
    if f.alternate() {
        f.options.flags |= 1 << (rt::Flag::SignAwareZeroPad as u32);

        if f.options.width.is_none() {
            f.options.width = Some((usize::BITS / 4) as usize + 2);
        }
    }
    f.options.flags |= 1 << (rt::Flag::Alternate as u32);

    let ret = LowerHex::fmt(&ptr_addr, f);

    f.options.width = old_width;
    f.options.flags = old_flags;

    ret
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Pointer for *mut T {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Pointer::fmt(&(*self as *const T), f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Pointer for &T {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Pointer::fmt(&(*self as *const T), f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Pointer for &mut T {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Pointer::fmt(&(&**self as *const T), f)
    }
}

// Implementation of Display/Debug for various core types

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Debug for *const T {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Pointer::fmt(self, f)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Debug for *mut T {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Pointer::fmt(self, f)
    }
}

macro_rules! peel {
    ($name:ident, $($other:ident,)*) => (tuple! { $($other,)* })
}

macro_rules! tuple {
    () => ();
    ( $($name:ident,)+ ) => (
        maybe_tuple_doc! {
            $($name)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($name:Debug),+> Debug for ($($name,)+) where last_type!($($name,)+): ?Sized {
                #[allow(non_snake_case, unused_assignments)]
                fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                    let mut builder = f.debug_tuple("");
                    let ($(ref $name,)+) = *self;
                    $(
                        builder.field(&$name);
                    )+

                    builder.finish()
                }
            }
        }
        peel! { $($name,)+ }
    )
}

macro_rules! maybe_tuple_doc {
    ($a:ident @ #[$meta:meta] $item:item) => {
        #[doc(fake_variadic)]
        #[doc = "This trait is implemented for tuples up to twelve items long."]
        #[$meta]
        $item
    };
    ($a:ident $($rest_a:ident)+ @ #[$meta:meta] $item:item) => {
        #[doc(hidden)]
        #[$meta]
        $item
    };
}

macro_rules! last_type {
    ($a:ident,) => { $a };
    ($a:ident, $($rest_a:ident,)+) => { last_type!($($rest_a,)+) };
}

tuple! { E, D, C, B, A, Z, Y, X, W, V, U, T, }

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Debug> Debug for [T] {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for () {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.pad("()")
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Debug for PhantomData<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "PhantomData<{}>", crate::any::type_name::<T>())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Copy + Debug> Debug for Cell<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Cell").field("value", &self.get()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Debug> Debug for RefCell<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut d = f.debug_struct("RefCell");
        match self.try_borrow() {
            Ok(borrow) => d.field("value", &borrow),
            Err(_) => d.field("value", &format_args!("<borrowed>")),
        };
        d.finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Debug> Debug for Ref<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Debug> Debug for RefMut<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&*(self.deref()), f)
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: ?Sized> Debug for UnsafeCell<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("UnsafeCell").finish_non_exhaustive()
    }
}

#[unstable(feature = "sync_unsafe_cell", issue = "95439")]
impl<T: ?Sized> Debug for SyncUnsafeCell<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("SyncUnsafeCell").finish_non_exhaustive()
    }
}

// If you expected tests to be here, look instead at the core/tests/fmt.rs file,
// it's a lot easier than creating all of the rt::Piece structures here.
// There are also tests in the alloc crate, for those that need allocations.
