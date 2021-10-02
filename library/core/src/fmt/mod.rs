//! Utilities for formatting and printing strings.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::cell::{Cell, Ref, RefCell, RefMut, UnsafeCell};
use crate::char::EscapeDebugExtArgs;
use crate::marker::PhantomData;
use crate::mem;
use crate::num::fmt as numfmt;
use crate::ops::Deref;
use crate::result;
use crate::str;

mod builders;
#[cfg(not(no_fp_fmt_parse))]
mod float;
#[cfg(no_fp_fmt_parse)]
mod nofloat;
mod num;

#[stable(feature = "fmt_flags_align", since = "1.28.0")]
/// Possible alignments returned by `Formatter::align`
#[derive(Debug)]
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

#[stable(feature = "debug_builders", since = "1.2.0")]
pub use self::builders::{DebugList, DebugMap, DebugSet, DebugStruct, DebugTuple};

#[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
#[doc(hidden)]
pub mod rt {
    pub mod v1;
}

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
/// assert_eq!(format!("{}", pythagorean_triple), "(3, 4, 5)");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub type Result = result::Result<(), Error>;

/// The error type which is returned from formatting a message into a stream.
///
/// This type does not support transmission of an error other than that an error
/// occurred. Any extra information must be arranged to be transmitted through
/// some other means.
///
/// An important thing to remember is that the type `fmt::Error` should not be
/// confused with [`std::io::Error`] or [`std::error::Error`], which you may also
/// have in scope.
///
/// [`std::io::Error`]: ../../std/io/struct.Error.html
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
    /// This function will return an instance of [`Error`] on error.
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
    /// writer(&mut buf, "hola").unwrap();
    /// assert_eq!(&buf, "hola");
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
    /// writer(&mut buf, 'a').unwrap();
    /// writer(&mut buf, 'b').unwrap();
    /// assert_eq!(&buf, "ab");
    /// ```
    #[stable(feature = "fmt_write_char", since = "1.1.0")]
    fn write_char(&mut self, c: char) -> Result {
        self.write_str(c.encode_utf8(&mut [0; 4]))
    }

    /// Glue for usage of the [`write!`] macro with implementors of this trait.
    ///
    /// This method should generally not be invoked manually, but rather through
    /// the [`write!`] macro itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::{Error, Write};
    ///
    /// fn writer<W: Write>(f: &mut W, s: &str) -> Result<(), Error> {
    ///     f.write_fmt(format_args!("{}", s))
    /// }
    ///
    /// let mut buf = String::new();
    /// writer(&mut buf, "world").unwrap();
    /// assert_eq!(&buf, "world");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn write_fmt(mut self: &mut Self, args: Arguments<'_>) -> Result {
        write(&mut self, args)
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
pub struct Formatter<'a> {
    flags: u32,
    fill: char,
    align: rt::v1::Alignment,
    width: Option<usize>,
    precision: Option<usize>,

    buf: &'a mut (dyn Write + 'a),
}

impl<'a> Formatter<'a> {
    /// Creates a new formatter with default settings.
    ///
    /// This can be used as a micro-optimization in cases where a full `Arguments`
    /// structure (as created by `format_args!`) is not necessary; `Arguments`
    /// is a little more expensive to use in simple formatting scenarios.
    ///
    /// Currently not intended for use outside of the standard library.
    #[unstable(feature = "fmt_internals", reason = "internal to standard library", issue = "none")]
    #[doc(hidden)]
    pub fn new(buf: &'a mut (dyn Write + 'a)) -> Formatter<'a> {
        Formatter {
            flags: 0,
            fill: ' ',
            align: rt::v1::Alignment::Unknown,
            width: None,
            precision: None,
            buf,
        }
    }
}

// NB. Argument is essentially an optimized partially applied formatting function,
// equivalent to `exists T.(&T, fn(&T, &mut Formatter<'_>) -> Result`.

extern "C" {
    type Opaque;
}

/// This struct represents the generic "argument" which is taken by the Xprintf
/// family of functions. It contains a function to format the given value. At
/// compile time it is ensured that the function and the value have the correct
/// types, and then this struct is used to canonicalize arguments to one type.
#[derive(Copy, Clone)]
#[allow(missing_debug_implementations)]
#[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
#[doc(hidden)]
pub struct ArgumentV1<'a> {
    value: &'a Opaque,
    formatter: fn(&Opaque, &mut Formatter<'_>) -> Result,
}

/// This struct represents the unsafety of constructing an `Arguments`.
/// It exists, rather than an unsafe function, in order to simplify the expansion
/// of `format_args!(..)` and reduce the scope of the `unsafe` block.
#[allow(missing_debug_implementations)]
#[doc(hidden)]
#[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
pub struct UnsafeArg {
    _private: (),
}

impl UnsafeArg {
    /// See documentation where `UnsafeArg` is required to know when it is safe to
    /// create and use `UnsafeArg`.
    #[doc(hidden)]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
    #[inline(always)]
    pub unsafe fn new() -> Self {
        Self { _private: () }
    }
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
#[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
static USIZE_MARKER: fn(&usize, &mut Formatter<'_>) -> Result = |ptr, _| {
    // SAFETY: ptr is a reference
    let _v: usize = unsafe { crate::ptr::read_volatile(ptr) };
    loop {}
};

impl<'a> ArgumentV1<'a> {
    #[doc(hidden)]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
    pub fn new<'b, T>(x: &'b T, f: fn(&T, &mut Formatter<'_>) -> Result) -> ArgumentV1<'b> {
        // SAFETY: `mem::transmute(x)` is safe because
        //     1. `&'b T` keeps the lifetime it originated with `'b`
        //              (so as to not have an unbounded lifetime)
        //     2. `&'b T` and `&'b Opaque` have the same memory layout
        //              (when `T` is `Sized`, as it is here)
        // `mem::transmute(f)` is safe since `fn(&T, &mut Formatter<'_>) -> Result`
        // and `fn(&Opaque, &mut Formatter<'_>) -> Result` have the same ABI
        // (as long as `T` is `Sized`)
        unsafe { ArgumentV1 { formatter: mem::transmute(f), value: mem::transmute(x) } }
    }

    #[doc(hidden)]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
    pub fn from_usize(x: &usize) -> ArgumentV1<'_> {
        ArgumentV1::new(x, USIZE_MARKER)
    }

    fn as_usize(&self) -> Option<usize> {
        if self.formatter as usize == USIZE_MARKER as usize {
            // SAFETY: The `formatter` field is only set to USIZE_MARKER if
            // the value is a usize, so this is safe
            Some(unsafe { *(self.value as *const _ as *const usize) })
        } else {
            None
        }
    }
}

// flags available in the v1 format of format_args
#[derive(Copy, Clone)]
enum FlagV1 {
    SignPlus,
    SignMinus,
    Alternate,
    SignAwareZeroPad,
    DebugLowerHex,
    DebugUpperHex,
}

impl<'a> Arguments<'a> {
    /// When using the format_args!() macro, this function is used to generate the
    /// Arguments structure.
    #[doc(hidden)]
    #[inline]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
    #[rustc_const_unstable(feature = "const_fmt_arguments_new", issue = "none")]
    pub const fn new_v1(pieces: &'a [&'static str], args: &'a [ArgumentV1<'a>]) -> Arguments<'a> {
        if pieces.len() < args.len() || pieces.len() > args.len() + 1 {
            panic!("invalid args");
        }
        Arguments { pieces, fmt: None, args }
    }

    /// This function is used to specify nonstandard formatting parameters.
    ///
    /// An `UnsafeArg` is required because the following invariants must be held
    /// in order for this function to be safe:
    /// 1. The `pieces` slice must be at least as long as `fmt`.
    /// 2. Every [`rt::v1::Argument::position`] value within `fmt` must be a
    ///    valid index of `args`.
    /// 3. Every [`Count::Param`] within `fmt` must contain a valid index of
    ///    `args`.
    #[cfg(not(bootstrap))]
    #[doc(hidden)]
    #[inline]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
    #[rustc_const_unstable(feature = "const_fmt_arguments_new", issue = "none")]
    pub const fn new_v1_formatted(
        pieces: &'a [&'static str],
        args: &'a [ArgumentV1<'a>],
        fmt: &'a [rt::v1::Argument],
        _unsafe_arg: UnsafeArg,
    ) -> Arguments<'a> {
        Arguments { pieces, fmt: Some(fmt), args }
    }

    #[cfg(bootstrap)]
    #[doc(hidden)]
    #[inline]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
    #[rustc_const_unstable(feature = "const_fmt_arguments_new", issue = "none")]
    pub const unsafe fn new_v1_formatted(
        pieces: &'a [&'static str],
        args: &'a [ArgumentV1<'a>],
        fmt: &'a [rt::v1::Argument],
    ) -> Arguments<'a> {
        Arguments { pieces, fmt: Some(fmt), args }
    }

    /// Estimates the length of the formatted text.
    ///
    /// This is intended to be used for setting initial `String` capacity
    /// when using `format!`. Note: this is neither the lower nor upper bound.
    #[doc(hidden)]
    #[inline]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]
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
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone)]
pub struct Arguments<'a> {
    // Format string pieces to print.
    pieces: &'a [&'static str],

    // Placeholder specs, or `None` if all specs are default (as in "{}{}").
    fmt: Option<&'a [rt::v1::Argument]>,

    // Dynamic arguments for interpolation, to be interleaved with string
    // pieces. (Every argument is preceded by a string piece.)
    args: &'a [ArgumentV1<'a>],
}

impl<'a> Arguments<'a> {
    /// Get the formatted string, if it has no arguments to be formatted.
    ///
    /// This can be used to avoid allocations in the most trivial case.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::fmt::Arguments;
    ///
    /// fn write_str(_: &str) { /* ... */ }
    ///
    /// fn write_fmt(args: &Arguments) {
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
    /// assert_eq!(format_args!("{}", 1).as_str(), None);
    /// ```
    #[stable(feature = "fmt_as_str", since = "1.52.0")]
    #[rustc_const_unstable(feature = "const_arguments_as_str", issue = "none")]
    #[inline]
    pub const fn as_str(&self) -> Option<&'static str> {
        match (self.pieces, self.args) {
            ([], []) => Some(""),
            ([s], []) => Some(s),
            _ => None,
        }
    }
}

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
/// standard library (`libstd`, `libcore`, `liballoc`, etc.) are not stable, and
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
/// assert_eq!(format!("The origin is: {:?}", origin), "The origin is: Point { x: 0, y: 0 }");
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
/// assert_eq!(format!("The origin is: {:?}", origin), "The origin is: Point { x: 0, y: 0 }");
/// ```
///
/// There are a number of helper methods on the [`Formatter`] struct to help you with manual
/// implementations, such as [`debug_struct`].
///
/// `Debug` implementations using either `derive` or the debug builder API
/// on [`Formatter`] support pretty-printing using the alternate flag: `{:#?}`.
///
/// [`debug_struct`]: Formatter::debug_struct
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
/// assert_eq!(format!("The origin is: {:#?}", origin),
/// "The origin is: Point {
///     x: 0,
///     y: 0,
/// }");
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
#[cfg_attr(not(bootstrap), rustc_trivial_field_reads)]
pub trait Debug {
    /// Formats the value using the given formatter.
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
    /// assert_eq!(format!("{:?}", position), "(1.987, 2.983)");
    ///
    /// assert_eq!(format!("{:#?}", position), "(
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
    #[allow_internal_unstable(core_intrinsics)]
    pub macro Debug($item:item) {
        /* compiler built-in */
    }
}
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(inline)]
pub use macros::Debug;

/// Format trait for an empty format, `{}`.
///
/// `Display` is similar to [`Debug`], but `Display` is for user-facing
/// output, and so cannot be derived.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
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
/// assert_eq!(format!("The origin is: {}", origin), "The origin is: (0, 0)");
/// ```
#[rustc_on_unimplemented(
    on(
        _Self = "std::path::Path",
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
    /// Formats the value using the given formatter.
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
    /// assert_eq!("(1.987, 2.983)",
    ///            format!("{}", Position { longitude: 1.987, latitude: 2.983, }));
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
/// assert_eq!(format!("{:o}", x), "52");
/// assert_eq!(format!("{:#o}", x), "0o52");
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
/// assert_eq!(format!("l as octal is: {:o}", l), "l as octal is: 11");
///
/// assert_eq!(format!("l as octal is: {:#06o}", l), "l as octal is: 0o0011");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Octal {
    /// Formats the value using the given formatter.
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
/// assert_eq!(format!("{:b}", x), "101010");
/// assert_eq!(format!("{:#b}", x), "0b101010");
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
/// assert_eq!(format!("l as binary is: {:b}", l), "l as binary is: 1101011");
///
/// assert_eq!(
///     format!("l as binary is: {:#032b}", l),
///     "l as binary is: 0b000000000000000000000001101011"
/// );
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Binary {
    /// Formats the value using the given formatter.
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
/// let x = 42; // 42 is '2a' in hex
///
/// assert_eq!(format!("{:x}", x), "2a");
/// assert_eq!(format!("{:#x}", x), "0x2a");
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
/// assert_eq!(format!("l as hex is: {:x}", l), "l as hex is: 9");
///
/// assert_eq!(format!("l as hex is: {:#010x}", l), "l as hex is: 0x00000009");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait LowerHex {
    /// Formats the value using the given formatter.
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
/// let x = 42; // 42 is '2A' in hex
///
/// assert_eq!(format!("{:X}", x), "2A");
/// assert_eq!(format!("{:#X}", x), "0x2A");
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
/// assert_eq!(format!("l as hex is: {:X}", l), "l as hex is: 7FFFFFFF");
///
/// assert_eq!(format!("l as hex is: {:#010X}", l), "l as hex is: 0x7FFFFFFF");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait UpperHex {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// `p` formatting.
///
/// The `Pointer` trait should format its output as a memory location. This is commonly presented
/// as hexadecimal.
///
/// For more information on formatters, see [the module-level documentation][module].
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
/// let address = format!("{:p}", x); // this produces something like '0x7f06092ac6d0'
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
/// println!("l is in memory here: {:p}", l);
///
/// let l_ptr = format!("{:018p}", l);
/// assert_eq!(l_ptr.len(), 18);
/// assert_eq!(&l_ptr[..2], "0x");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Pointer"]
pub trait Pointer {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "pointer_trait_fmt"]
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
/// assert_eq!(format!("{:e}", x), "4.2e1");
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
///     format!("l in scientific notation is: {:e}", l),
///     "l in scientific notation is: 1e2"
/// );
///
/// assert_eq!(
///     format!("l in scientific notation is: {:05e}", l),
///     "l in scientific notation is: 001e2"
/// );
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait LowerExp {
    /// Formats the value using the given formatter.
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
/// assert_eq!(format!("{:E}", x), "4.2E1");
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
///     format!("l in scientific notation is: {:E}", l),
///     "l in scientific notation is: 1E2"
/// );
///
/// assert_eq!(
///     format!("l in scientific notation is: {:05E}", l),
///     "l in scientific notation is: 001E2"
/// );
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait UpperExp {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result;
}

/// The `write` function takes an output stream, and an `Arguments` struct
/// that can be precompiled with the `format_args!` macro.
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
    let mut formatter = Formatter::new(output);
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
                (arg.formatter)(arg.value, &mut formatter)?;
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

unsafe fn run(fmt: &mut Formatter<'_>, arg: &rt::v1::Argument, args: &[ArgumentV1<'_>]) -> Result {
    fmt.fill = arg.format.fill;
    fmt.align = arg.format.align;
    fmt.flags = arg.format.flags;
    // SAFETY: arg and args come from the same Arguments,
    // which guarantees the indexes are always within bounds.
    unsafe {
        fmt.width = getcount(args, &arg.format.width);
        fmt.precision = getcount(args, &arg.format.precision);
    }

    // Extract the correct argument
    debug_assert!(arg.position < args.len());
    // SAFETY: arg and args come from the same Arguments,
    // which guarantees its index is always within bounds.
    let value = unsafe { args.get_unchecked(arg.position) };

    // Then actually do some printing
    (value.formatter)(value.value, fmt)
}

unsafe fn getcount(args: &[ArgumentV1<'_>], cnt: &rt::v1::Count) -> Option<usize> {
    match *cnt {
        rt::v1::Count::Is(n) => Some(n),
        rt::v1::Count::Implied => None,
        rt::v1::Count::Param(i) => {
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

    /// Write this post padding.
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
            flags: self.flags,
            fill: self.fill,
            align: self.align,
            width: self.width,
            precision: self.precision,
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
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         // We need to remove "-" from the number output.
    ///         let tmp = self.nb.abs().to_string();
    ///
    ///         formatter.pad_integral(self.nb >= 0, "Foo ", &tmp)
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{}", Foo::new(2)), "2");
    /// assert_eq!(&format!("{}", Foo::new(-1)), "-1");
    /// assert_eq!(&format!("{}", Foo::new(0)), "0");
    /// assert_eq!(&format!("{:#}", Foo::new(-1)), "-Foo 1");
    /// assert_eq!(&format!("{:0>#8}", Foo::new(-1)), "00-Foo 1");
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
        match self.width {
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
                let old_fill = crate::mem::replace(&mut self.fill, '0');
                let old_align = crate::mem::replace(&mut self.align, rt::v1::Alignment::Right);
                write_prefix(self, sign, prefix)?;
                let post_padding = self.padding(min - width, rt::v1::Alignment::Right)?;
                self.buf.write_str(buf)?;
                post_padding.write(self)?;
                self.fill = old_fill;
                self.align = old_align;
                Ok(())
            }
            // Otherwise, the sign and prefix goes after the padding
            Some(min) => {
                let post_padding = self.padding(min - width, rt::v1::Alignment::Right)?;
                write_prefix(self, sign, prefix)?;
                self.buf.write_str(buf)?;
                post_padding.write(self)
            }
        }
    }

    /// This function takes a string slice and emits it to the internal buffer
    /// after applying the relevant formatting flags specified. The flags
    /// recognized for generic strings are:
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
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         formatter.pad("Foo")
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{:<4}", Foo), "Foo ");
    /// assert_eq!(&format!("{:0>4}", Foo), "0Foo");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pad(&mut self, s: &str) -> Result {
        // Make sure there's a fast path up front
        if self.width.is_none() && self.precision.is_none() {
            return self.buf.write_str(s);
        }
        // The `precision` field can be interpreted as a `max-width` for the
        // string being formatted.
        let s = if let Some(max) = self.precision {
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
        match self.width {
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
                    let align = rt::v1::Alignment::Left;
                    let post_padding = self.padding(width - chars_count, align)?;
                    self.buf.write_str(s)?;
                    post_padding.write(self)
                }
            }
        }
    }

    /// Write the pre-padding and return the unwritten post-padding. Callers are
    /// responsible for ensuring post-padding is written after the thing that is
    /// being padded.
    pub(crate) fn padding(
        &mut self,
        padding: usize,
        default: rt::v1::Alignment,
    ) -> result::Result<PostPadding, Error> {
        let align = match self.align {
            rt::v1::Alignment::Unknown => default,
            _ => self.align,
        };

        let (pre_pad, post_pad) = match align {
            rt::v1::Alignment::Left => (0, padding),
            rt::v1::Alignment::Right | rt::v1::Alignment::Unknown => (padding, 0),
            rt::v1::Alignment::Center => (padding / 2, (padding + 1) / 2),
        };

        for _ in 0..pre_pad {
            self.buf.write_char(self.fill)?;
        }

        Ok(PostPadding::new(self.fill, post_pad))
    }

    /// Takes the formatted parts and applies the padding.
    /// Assumes that the caller already has rendered the parts with required precision,
    /// so that `self.precision` can be ignored.
    fn pad_formatted_parts(&mut self, formatted: &numfmt::Formatted<'_>) -> Result {
        if let Some(mut width) = self.width {
            // for the sign-aware zero padding, we render the sign first and
            // behave as if we had no sign from the beginning.
            let mut formatted = formatted.clone();
            let old_fill = self.fill;
            let old_align = self.align;
            let mut align = old_align;
            if self.sign_aware_zero_pad() {
                // a sign always goes first
                let sign = formatted.sign;
                self.buf.write_str(sign)?;

                // remove the sign from the formatted parts
                formatted.sign = "";
                width = width.saturating_sub(sign.len());
                align = rt::v1::Alignment::Right;
                self.fill = '0';
                self.align = rt::v1::Alignment::Right;
            }

            // remaining parts go through the ordinary padding process.
            let len = formatted.len();
            let ret = if width <= len {
                // no padding
                self.write_formatted_parts(&formatted)
            } else {
                let post_padding = self.padding(width - len, align)?;
                self.write_formatted_parts(&formatted)?;
                post_padding.write(self)
            };
            self.fill = old_fill;
            self.align = old_align;
            ret
        } else {
            // this is the common case and we take a shortcut
            self.write_formatted_parts(formatted)
        }
    }

    fn write_formatted_parts(&mut self, formatted: &numfmt::Formatted<'_>) -> Result {
        fn write_bytes(buf: &mut dyn Write, s: &[u8]) -> Result {
            // SAFETY: This is used for `numfmt::Part::Num` and `numfmt::Part::Copy`.
            // It's safe to use for `numfmt::Part::Num` since every char `c` is between
            // `b'0'` and `b'9'`, which means `s` is valid UTF-8.
            // It's also probably safe in practice to use for `numfmt::Part::Copy(buf)`
            // since `buf` should be plain ASCII, but it's possible for someone to pass
            // in a bad value for `buf` into `numfmt::to_shortest_str` since it is a
            // public function.
            // FIXME: Determine whether this could result in UB.
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
                    write_bytes(self.buf, &s[..len])?;
                }
                numfmt::Part::Copy(buf) => {
                    write_bytes(self.buf, buf)?;
                }
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
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         formatter.write_str("Foo")
    ///         // This is equivalent to:
    ///         // write!(formatter, "Foo")
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{}", Foo), "Foo");
    /// assert_eq!(&format!("{:0>8}", Foo), "Foo");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn write_str(&mut self, data: &str) -> Result {
        self.buf.write_str(data)
    }

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
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         formatter.write_fmt(format_args!("Foo {}", self.0))
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{}", Foo(-1)), "Foo -1");
    /// assert_eq!(&format!("{:0>8}", Foo(2)), "Foo 2");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn write_fmt(&mut self, fmt: Arguments<'_>) -> Result {
        write(self.buf, fmt)
    }

    /// Flags for formatting
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(
        since = "1.24.0",
        reason = "use the `sign_plus`, `sign_minus`, `alternate`, \
                  or `sign_aware_zero_pad` methods instead"
    )]
    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// Character used as 'fill' whenever there is alignment.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo;
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         let c = formatter.fill();
    ///         if let Some(width) = formatter.width() {
    ///             for _ in 0..width {
    ///                 write!(formatter, "{}", c)?;
    ///             }
    ///             Ok(())
    ///         } else {
    ///             write!(formatter, "{}", c)
    ///         }
    ///     }
    /// }
    ///
    /// // We set alignment to the right with ">".
    /// assert_eq!(&format!("{:G>3}", Foo), "GGG");
    /// assert_eq!(&format!("{:t>6}", Foo), "tttttt");
    /// ```
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn fill(&self) -> char {
        self.fill
    }

    /// Flag indicating what form of alignment was requested.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate core;
    ///
    /// use std::fmt::{self, Alignment};
    ///
    /// struct Foo;
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         let s = if let Some(s) = formatter.align() {
    ///             match s {
    ///                 Alignment::Left    => "left",
    ///                 Alignment::Right   => "right",
    ///                 Alignment::Center  => "center",
    ///             }
    ///         } else {
    ///             "into the void"
    ///         };
    ///         write!(formatter, "{}", s)
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{:<}", Foo), "left");
    /// assert_eq!(&format!("{:>}", Foo), "right");
    /// assert_eq!(&format!("{:^}", Foo), "center");
    /// assert_eq!(&format!("{}", Foo), "into the void");
    /// ```
    #[stable(feature = "fmt_flags_align", since = "1.28.0")]
    pub fn align(&self) -> Option<Alignment> {
        match self.align {
            rt::v1::Alignment::Left => Some(Alignment::Left),
            rt::v1::Alignment::Right => Some(Alignment::Right),
            rt::v1::Alignment::Center => Some(Alignment::Center),
            rt::v1::Alignment::Unknown => None,
        }
    }

    /// Optionally specified integer width that the output should be.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo(i32);
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         if let Some(width) = formatter.width() {
    ///             // If we received a width, we use it
    ///             write!(formatter, "{:width$}", &format!("Foo({})", self.0), width = width)
    ///         } else {
    ///             // Otherwise we do nothing special
    ///             write!(formatter, "Foo({})", self.0)
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{:10}", Foo(23)), "Foo(23)   ");
    /// assert_eq!(&format!("{}", Foo(23)), "Foo(23)");
    /// ```
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn width(&self) -> Option<usize> {
        self.width
    }

    /// Optionally specified precision for numeric types. Alternatively, the
    /// maximum width for string types.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// struct Foo(f32);
    ///
    /// impl fmt::Display for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
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
    /// assert_eq!(&format!("{:.4}", Foo(23.2)), "Foo(23.2000)");
    /// assert_eq!(&format!("{}", Foo(23.2)), "Foo(23.20)");
    /// ```
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn precision(&self) -> Option<usize> {
        self.precision
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
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         if formatter.sign_plus() {
    ///             write!(formatter,
    ///                    "Foo({}{})",
    ///                    if self.0 < 0 { '-' } else { '+' },
    ///                    self.0)
    ///         } else {
    ///             write!(formatter, "Foo({})", self.0)
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{:+}", Foo(23)), "Foo(+23)");
    /// assert_eq!(&format!("{}", Foo(23)), "Foo(23)");
    /// ```
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn sign_plus(&self) -> bool {
        self.flags & (1 << FlagV1::SignPlus as u32) != 0
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
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         if formatter.sign_minus() {
    ///             // You want a minus sign? Have one!
    ///             write!(formatter, "-Foo({})", self.0)
    ///         } else {
    ///             write!(formatter, "Foo({})", self.0)
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{:-}", Foo(23)), "-Foo(23)");
    /// assert_eq!(&format!("{}", Foo(23)), "Foo(23)");
    /// ```
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn sign_minus(&self) -> bool {
        self.flags & (1 << FlagV1::SignMinus as u32) != 0
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
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         if formatter.alternate() {
    ///             write!(formatter, "Foo({})", self.0)
    ///         } else {
    ///             write!(formatter, "{}", self.0)
    ///         }
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{:#}", Foo(23)), "Foo(23)");
    /// assert_eq!(&format!("{}", Foo(23)), "23");
    /// ```
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn alternate(&self) -> bool {
        self.flags & (1 << FlagV1::Alternate as u32) != 0
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
    ///     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    ///         assert!(formatter.sign_aware_zero_pad());
    ///         assert_eq!(formatter.width(), Some(4));
    ///         // We ignore the formatter's options.
    ///         write!(formatter, "{}", self.0)
    ///     }
    /// }
    ///
    /// assert_eq!(&format!("{:04}", Foo(23)), "23");
    /// ```
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn sign_aware_zero_pad(&self) -> bool {
        self.flags & (1 << FlagV1::SignAwareZeroPad as u32) != 0
    }

    // FIXME: Decide what public API we want for these two flags.
    // https://github.com/rust-lang/rust/issues/48584
    fn debug_lower_hex(&self) -> bool {
        self.flags & (1 << FlagV1::DebugLowerHex as u32) != 0
    }

    fn debug_upper_hex(&self) -> bool {
        self.flags & (1 << FlagV1::DebugUpperHex as u32) != 0
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
    ///     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
    ///     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
    ///     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
    ///     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
    /// struct Arm<'a, L: 'a, R: 'a>(&'a (L, R));
    /// struct Table<'a, K: 'a, V: 'a>(&'a [(K, V)], V);
    ///
    /// impl<'a, L, R> fmt::Debug for Arm<'a, L, R>
    /// where
    ///     L: 'a + fmt::Debug, R: 'a + fmt::Debug
    /// {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
    ///     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
    ///     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
}

#[stable(since = "1.2.0", feature = "formatter_write")]
impl Write for Formatter<'_> {
    fn write_str(&mut self, s: &str) -> Result {
        self.buf.write_str(s)
    }

    fn write_char(&mut self, c: char) -> Result {
        self.buf.write_char(c)
    }

    fn write_fmt(&mut self, args: Arguments<'_>) -> Result {
        write(self.buf, args)
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
    fn fmt(&self, _: &mut Formatter<'_>) -> Result {
        *self
    }
}

#[unstable(feature = "never_type", issue = "35121")]
impl Display for ! {
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
        let mut from = 0;
        for (i, c) in self.char_indices() {
            let esc = c.escape_debug_ext(EscapeDebugExtArgs {
                escape_grapheme_extended: true,
                escape_single_quote: false,
                escape_double_quote: true,
            });
            // If char needs escaping, flush backlog so far and write, else skip
            if esc.len() != 1 {
                f.write_str(&self[from..i])?;
                for c in esc {
                    f.write_char(c)?;
                }
                from = i + c.len_utf8();
            }
        }
        f.write_str(&self[from..])?;
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
        for c in self.escape_debug_ext(EscapeDebugExtArgs {
            escape_grapheme_extended: true,
            escape_single_quote: true,
            escape_double_quote: false,
        }) {
            f.write_char(c)?
        }
        f.write_char('\'')
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Display for char {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if f.width.is_none() && f.precision.is_none() {
            f.write_char(*self)
        } else {
            f.pad(self.encode_utf8(&mut [0; 4]))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Pointer for *const T {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let old_width = f.width;
        let old_flags = f.flags;

        // The alternate flag is already treated by LowerHex as being special-
        // it denotes whether to prefix with 0x. We use it to work out whether
        // or not to zero extend, and then unconditionally set it to get the
        // prefix.
        if f.alternate() {
            f.flags |= 1 << (FlagV1::SignAwareZeroPad as u32);

            if f.width.is_none() {
                f.width = Some((usize::BITS / 4) as usize + 2);
            }
        }
        f.flags |= 1 << (FlagV1::Alternate as u32);

        let ret = LowerHex::fmt(&(*self as *const () as usize), f);

        f.width = old_width;
        f.flags = old_flags;

        ret
    }
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
        peel! { $($name,)+ }
    )
}

macro_rules! last_type {
    ($a:ident,) => { $a };
    ($a:ident, $($rest_a:ident,)+) => { last_type!($($rest_a,)+) };
}

tuple! { T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, }

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
        f.debug_struct("PhantomData").finish()
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
        match self.try_borrow() {
            Ok(borrow) => f.debug_struct("RefCell").field("value", &borrow).finish(),
            Err(_) => {
                // The RefCell is mutably borrowed so we can't look at its value
                // here. Show a placeholder instead.
                struct BorrowedPlaceholder;

                impl Debug for BorrowedPlaceholder {
                    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                        f.write_str("<borrowed>")
                    }
                }

                f.debug_struct("RefCell").field("value", &BorrowedPlaceholder).finish()
            }
        }
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

// If you expected tests to be here, look instead at the core/tests/fmt.rs file,
// it's a lot easier than creating all of the rt::Piece structures here.
// There are also tests in the alloc crate, for those that need allocations.
