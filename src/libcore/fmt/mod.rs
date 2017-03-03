// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utilities for formatting and printing strings.

#![stable(feature = "rust1", since = "1.0.0")]

use cell::{UnsafeCell, Cell, RefCell, Ref, RefMut};
use marker::PhantomData;
use mem;
use num::flt2dec;
use ops::Deref;
use result;
use slice;
use str;

mod float;
mod num;
mod builders;

#[unstable(feature = "fmt_flags_align", issue = "27726")]
/// Possible alignments returned by `Formatter::align`
#[derive(Debug)]
pub enum Alignment {
    /// Indication that contents should be left-aligned.
    Left,
    /// Indication that contents should be right-aligned.
    Right,
    /// Indication that contents should be center-aligned.
    Center,
    /// No alignment was requested.
    Unknown,
}

#[stable(feature = "debug_builders", since = "1.2.0")]
pub use self::builders::{DebugStruct, DebugTuple, DebugSet, DebugList, DebugMap};

#[unstable(feature = "fmt_internals", reason = "internal to format_args!",
           issue = "0")]
#[doc(hidden)]
pub mod rt {
    pub mod v1;
}

#[stable(feature = "rust1", since = "1.0.0")]
/// The type returned by formatter methods.
pub type Result = result::Result<(), Error>;

/// The error type which is returned from formatting a message into a stream.
///
/// This type does not support transmission of an error other than that an error
/// occurred. Any extra information must be arranged to be transmitted through
/// some other means.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Error;

/// A collection of methods that are required to format a message into a stream.
///
/// This trait is the type which this modules requires when formatting
/// information. This is similar to the standard library's [`io::Write`] trait,
/// but it is only intended for use in libcore.
///
/// This trait should generally not be implemented by consumers of the standard
/// library. The [`write!`] macro accepts an instance of [`io::Write`], and the
/// [`io::Write`] trait is favored over implementing this trait.
///
/// [`write!`]: ../../std/macro.write.html
/// [`io::Write`]: ../../std/io/trait.Write.html
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Write {
    /// Writes a slice of bytes into this writer, returning whether the write
    /// succeeded.
    ///
    /// This method can only succeed if the entire byte slice was successfully
    /// written, and this method will not return until all data has been
    /// written or an error occurs.
    ///
    /// # Errors
    ///
    /// This function will return an instance of [`Error`] on error.
    ///
    /// [`Error`]: struct.Error.html
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
    /// [`char`]: ../../std/primitive.char.html
    /// [`Error`]: struct.Error.html
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
    /// [`write!`]: ../../std/macro.write.html
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
    fn write_fmt(&mut self, args: Arguments) -> Result {
        // This Adapter is needed to allow `self` (of type `&mut
        // Self`) to be cast to a Write (below) without
        // requiring a `Sized` bound.
        struct Adapter<'a,T: ?Sized +'a>(&'a mut T);

        impl<'a, T: ?Sized> Write for Adapter<'a, T>
            where T: Write
        {
            fn write_str(&mut self, s: &str) -> Result {
                self.0.write_str(s)
            }

            fn write_char(&mut self, c: char) -> Result {
                self.0.write_char(c)
            }

            fn write_fmt(&mut self, args: Arguments) -> Result {
                self.0.write_fmt(args)
            }
        }

        write(&mut Adapter(self), args)
    }
}

#[stable(feature = "fmt_write_blanket_impl", since = "1.4.0")]
impl<'a, W: Write + ?Sized> Write for &'a mut W {
    fn write_str(&mut self, s: &str) -> Result {
        (**self).write_str(s)
    }

    fn write_char(&mut self, c: char) -> Result {
        (**self).write_char(c)
    }

    fn write_fmt(&mut self, args: Arguments) -> Result {
        (**self).write_fmt(args)
    }
}

/// A struct to represent both where to emit formatting strings to and how they
/// should be formatted. A mutable version of this is passed to all formatting
/// traits.
#[allow(missing_debug_implementations)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Formatter<'a> {
    flags: u32,
    fill: char,
    align: rt::v1::Alignment,
    width: Option<usize>,
    precision: Option<usize>,

    buf: &'a mut (Write+'a),
    curarg: slice::Iter<'a, ArgumentV1<'a>>,
    args: &'a [ArgumentV1<'a>],
}

// NB. Argument is essentially an optimized partially applied formatting function,
// equivalent to `exists T.(&T, fn(&T, &mut Formatter) -> Result`.

struct Void {
    _priv: (),
}

/// This struct represents the generic "argument" which is taken by the Xprintf
/// family of functions. It contains a function to format the given value. At
/// compile time it is ensured that the function and the value have the correct
/// types, and then this struct is used to canonicalize arguments to one type.
#[derive(Copy)]
#[allow(missing_debug_implementations)]
#[unstable(feature = "fmt_internals", reason = "internal to format_args!",
           issue = "0")]
#[doc(hidden)]
pub struct ArgumentV1<'a> {
    value: &'a Void,
    formatter: fn(&Void, &mut Formatter) -> Result,
}

#[unstable(feature = "fmt_internals", reason = "internal to format_args!",
           issue = "0")]
impl<'a> Clone for ArgumentV1<'a> {
    fn clone(&self) -> ArgumentV1<'a> {
        *self
    }
}

impl<'a> ArgumentV1<'a> {
    #[inline(never)]
    fn show_usize(x: &usize, f: &mut Formatter) -> Result {
        Display::fmt(x, f)
    }

    #[doc(hidden)]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!",
               issue = "0")]
    pub fn new<'b, T>(x: &'b T,
                      f: fn(&T, &mut Formatter) -> Result) -> ArgumentV1<'b> {
        unsafe {
            ArgumentV1 {
                formatter: mem::transmute(f),
                value: mem::transmute(x)
            }
        }
    }

    #[doc(hidden)]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!",
               issue = "0")]
    pub fn from_usize(x: &usize) -> ArgumentV1 {
        ArgumentV1::new(x, ArgumentV1::show_usize)
    }

    fn as_usize(&self) -> Option<usize> {
        if self.formatter as usize == ArgumentV1::show_usize as usize {
            Some(unsafe { *(self.value as *const _ as *const usize) })
        } else {
            None
        }
    }
}

// flags available in the v1 format of format_args
#[derive(Copy, Clone)]
#[allow(dead_code)] // SignMinus isn't currently used
enum FlagV1 { SignPlus, SignMinus, Alternate, SignAwareZeroPad, }

impl<'a> Arguments<'a> {
    /// When using the format_args!() macro, this function is used to generate the
    /// Arguments structure.
    #[doc(hidden)] #[inline]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!",
               issue = "0")]
    pub fn new_v1(pieces: &'a [&'a str],
                  args: &'a [ArgumentV1<'a>]) -> Arguments<'a> {
        Arguments {
            pieces: pieces,
            fmt: None,
            args: args
        }
    }

    /// This function is used to specify nonstandard formatting parameters.
    /// The `pieces` array must be at least as long as `fmt` to construct
    /// a valid Arguments structure. Also, any `Count` within `fmt` that is
    /// `CountIsParam` or `CountIsNextParam` has to point to an argument
    /// created with `argumentusize`. However, failing to do so doesn't cause
    /// unsafety, but will ignore invalid .
    #[doc(hidden)] #[inline]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!",
               issue = "0")]
    pub fn new_v1_formatted(pieces: &'a [&'a str],
                            args: &'a [ArgumentV1<'a>],
                            fmt: &'a [rt::v1::Argument]) -> Arguments<'a> {
        Arguments {
            pieces: pieces,
            fmt: Some(fmt),
            args: args
        }
    }

    /// Estimates the length of the formatted text.
    ///
    /// This is intended to be used for setting initial `String` capacity
    /// when using `format!`. Note: this is neither the lower nor upper bound.
    #[doc(hidden)] #[inline]
    #[unstable(feature = "fmt_internals", reason = "internal to format_args!",
               issue = "0")]
    pub fn estimated_capacity(&self) -> usize {
        let pieces_length: usize = self.pieces.iter()
            .map(|x| x.len()).sum();

        if self.args.is_empty() {
            pieces_length
        } else if self.pieces[0] == "" && pieces_length < 16 {
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
/// The [`format_args!`] macro will safely create an instance of this structure
/// and pass it to a function or closure, passed as the first argument. The
/// macro validates the format string at compile-time so usage of the [`write`]
/// and [`format`] functions can be safely performed.
///
/// [`format_args!`]: ../../std/macro.format_args.html
/// [`format`]: ../../std/fmt/fn.format.html
/// [`write`]: ../../std/fmt/fn.write.html
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Copy, Clone)]
pub struct Arguments<'a> {
    // Format string pieces to print.
    pieces: &'a [&'a str],

    // Placeholder specs, or `None` if all specs are default (as in "{}{}").
    fmt: Option<&'a [rt::v1::Argument]>,

    // Dynamic arguments for interpolation, to be interleaved with string
    // pieces. (Every argument is preceded by a string piece.)
    args: &'a [ArgumentV1<'a>],
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Debug for Arguments<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result {
        Display::fmt(self, fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Display for Arguments<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result {
        write(fmt.buf, *self)
    }
}

/// Format trait for the `?` character.
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
/// println!("The origin is: {:?}", origin);
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
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         write!(f, "Point {{ x: {}, y: {} }}", self.x, self.y)
///     }
/// }
///
/// let origin = Point { x: 0, y: 0 };
///
/// println!("The origin is: {:?}", origin);
/// ```
///
/// This outputs:
///
/// ```text
/// The origin is: Point { x: 0, y: 0 }
/// ```
///
/// There are a number of `debug_*` methods on `Formatter` to help you with manual
/// implementations, such as [`debug_struct`][debug_struct].
///
/// `Debug` implementations using either `derive` or the debug builder API
/// on `Formatter` support pretty printing using the alternate flag: `{:#?}`.
///
/// [debug_struct]: ../../std/fmt/struct.Formatter.html#method.debug_struct
///
/// Pretty printing with `#?`:
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
/// println!("The origin is: {:#?}", origin);
/// ```
///
/// This outputs:
///
/// ```text
/// The origin is: Point {
///     x: 0,
///     y: 0
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "`{Self}` cannot be formatted using `:?`; if it is \
                            defined in your crate, add `#[derive(Debug)]` or \
                            manually implement it"]
#[lang = "debug_trait"]
pub trait Debug {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter) -> Result;
}

/// Format trait for an empty format, `{}`.
///
/// `Display` is similar to [`Debug`][debug], but `Display` is for user-facing
/// output, and so cannot be derived.
///
/// [debug]: trait.Debug.html
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
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         write!(f, "({}, {})", self.x, self.y)
///     }
/// }
///
/// let origin = Point { x: 0, y: 0 };
///
/// println!("The origin is: {}", origin);
/// ```
#[rustc_on_unimplemented = "`{Self}` cannot be formatted with the default \
                            formatter; try using `:?` instead if you are using \
                            a format string"]
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
    ///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    ///         write!(f, "({}, {})", self.longitude, self.latitude)
    ///     }
    /// }
    ///
    /// assert_eq!("(1.987, 2.983)".to_owned(),
    ///            format!("{}", Position { longitude: 1.987, latitude: 2.983, }));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter) -> Result;
}

/// Format trait for the `o` character.
///
/// The `Octal` trait should format its output as a number in base-8.
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
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         let val = self.0;
///
///         write!(f, "{:o}", val) // delegate to i32's implementation
///     }
/// }
///
/// let l = Length(9);
///
/// println!("l as octal is: {:o}", l);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Octal {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter) -> Result;
}

/// Format trait for the `b` character.
///
/// The `Binary` trait should format its output as a number in binary.
///
/// The alternate flag, `#`, adds a `0b` in front of the output.
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
/// let x = 42; // 42 is '101010' in binary
///
/// assert_eq!(format!("{:b}", x), "101010");
/// assert_eq!(format!("{:#b}", x), "0b101010");
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
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         let val = self.0;
///
///         write!(f, "{:b}", val) // delegate to i32's implementation
///     }
/// }
///
/// let l = Length(107);
///
/// println!("l as binary is: {:b}", l);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Binary {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter) -> Result;
}

/// Format trait for the `x` character.
///
/// The `LowerHex` trait should format its output as a number in hexadecimal, with `a` through `f`
/// in lower case.
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
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         let val = self.0;
///
///         write!(f, "{:x}", val) // delegate to i32's implementation
///     }
/// }
///
/// let l = Length(9);
///
/// println!("l as hex is: {:x}", l);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait LowerHex {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter) -> Result;
}

/// Format trait for the `X` character.
///
/// The `UpperHex` trait should format its output as a number in hexadecimal, with `A` through `F`
/// in upper case.
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
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         let val = self.0;
///
///         write!(f, "{:X}", val) // delegate to i32's implementation
///     }
/// }
///
/// let l = Length(9);
///
/// println!("l as hex is: {:X}", l);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait UpperHex {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter) -> Result;
}

/// Format trait for the `p` character.
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
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         // use `as` to convert to a `*const T`, which implements Pointer, which we can use
///
///         write!(f, "{:p}", self as *const Length)
///     }
/// }
///
/// let l = Length(42);
///
/// println!("l is in memory here: {:p}", l);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Pointer {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter) -> Result;
}

/// Format trait for the `e` character.
///
/// The `LowerExp` trait should format its output in scientific notation with a lower-case `e`.
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
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         let val = self.0;
///         write!(f, "{}e1", val / 10)
///     }
/// }
///
/// let l = Length(100);
///
/// println!("l in scientific notation is: {:e}", l);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait LowerExp {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter) -> Result;
}

/// Format trait for the `E` character.
///
/// The `UpperExp` trait should format its output in scientific notation with an upper-case `E`.
///
/// For more information on formatters, see [the module-level documentation][module].
///
/// [module]: ../../std/fmt/index.html
///
/// # Examples
///
/// Basic usage with `f32`:
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
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         let val = self.0;
///         write!(f, "{}E1", val / 10)
///     }
/// }
///
/// let l = Length(100);
///
/// println!("l in scientific notation is: {:E}", l);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait UpperExp {
    /// Formats the value using the given formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fmt(&self, f: &mut Formatter) -> Result;
}

/// The `write` function takes an output stream, a precompiled format string,
/// and a list of arguments. The arguments will be formatted according to the
/// specified format string into the output stream provided.
///
/// # Arguments
///
///   * output - the buffer to write output to
///   * args - the precompiled arguments generated by `format_args!`
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
/// Please note that using [`write!`] might be preferrable. Example:
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
/// [`write!`]: ../../std/macro.write.html
#[stable(feature = "rust1", since = "1.0.0")]
pub fn write(output: &mut Write, args: Arguments) -> Result {
    let mut formatter = Formatter {
        flags: 0,
        width: None,
        precision: None,
        buf: output,
        align: rt::v1::Alignment::Unknown,
        fill: ' ',
        args: args.args,
        curarg: args.args.iter(),
    };

    let mut pieces = args.pieces.iter();

    match args.fmt {
        None => {
            // We can use default formatting parameters for all arguments.
            for (arg, piece) in args.args.iter().zip(pieces.by_ref()) {
                formatter.buf.write_str(*piece)?;
                (arg.formatter)(arg.value, &mut formatter)?;
            }
        }
        Some(fmt) => {
            // Every spec has a corresponding argument that is preceded by
            // a string piece.
            for (arg, piece) in fmt.iter().zip(pieces.by_ref()) {
                formatter.buf.write_str(*piece)?;
                formatter.run(arg)?;
            }
        }
    }

    // There can be only one trailing string piece left.
    if let Some(piece) = pieces.next() {
        formatter.buf.write_str(*piece)?;
    }

    Ok(())
}

impl<'a> Formatter<'a> {
    // First up is the collection of functions used to execute a format string
    // at runtime. This consumes all of the compile-time statics generated by
    // the format! syntax extension.
    fn run(&mut self, arg: &rt::v1::Argument) -> Result {
        // Fill in the format parameters into the formatter
        self.fill = arg.format.fill;
        self.align = arg.format.align;
        self.flags = arg.format.flags;
        self.width = self.getcount(&arg.format.width);
        self.precision = self.getcount(&arg.format.precision);

        // Extract the correct argument
        let value = match arg.position {
            rt::v1::Position::Next => { *self.curarg.next().unwrap() }
            rt::v1::Position::At(i) => self.args[i],
        };

        // Then actually do some printing
        (value.formatter)(value.value, self)
    }

    fn getcount(&mut self, cnt: &rt::v1::Count) -> Option<usize> {
        match *cnt {
            rt::v1::Count::Is(n) => Some(n),
            rt::v1::Count::Implied => None,
            rt::v1::Count::Param(i) => {
                self.args[i].as_usize()
            }
            rt::v1::Count::NextParam => {
                self.curarg.next().and_then(|arg| arg.as_usize())
            }
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pad_integral(&mut self,
                        is_nonnegative: bool,
                        prefix: &str,
                        buf: &str)
                        -> Result {
        let mut width = buf.len();

        let mut sign = None;
        if !is_nonnegative {
            sign = Some('-'); width += 1;
        } else if self.sign_plus() {
            sign = Some('+'); width += 1;
        }

        let mut prefixed = false;
        if self.alternate() {
            prefixed = true; width += prefix.chars().count();
        }

        // Writes the sign if it exists, and then the prefix if it was requested
        let write_prefix = |f: &mut Formatter| {
            if let Some(c) = sign {
                f.buf.write_str(c.encode_utf8(&mut [0; 4]))?;
            }
            if prefixed { f.buf.write_str(prefix) }
            else { Ok(()) }
        };

        // The `width` field is more of a `min-width` parameter at this point.
        match self.width {
            // If there's no minimum length requirements then we can just
            // write the bytes.
            None => {
                write_prefix(self)?; self.buf.write_str(buf)
            }
            // Check if we're over the minimum width, if so then we can also
            // just write the bytes.
            Some(min) if width >= min => {
                write_prefix(self)?; self.buf.write_str(buf)
            }
            // The sign and prefix goes before the padding if the fill character
            // is zero
            Some(min) if self.sign_aware_zero_pad() => {
                self.fill = '0';
                self.align = rt::v1::Alignment::Right;
                write_prefix(self)?;
                self.with_padding(min - width, rt::v1::Alignment::Right, |f| {
                    f.buf.write_str(buf)
                })
            }
            // Otherwise, the sign and prefix goes after the padding
            Some(min) => {
                self.with_padding(min - width, rt::v1::Alignment::Right, |f| {
                    write_prefix(f)?; f.buf.write_str(buf)
                })
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
            if let Some((i, _)) = s.char_indices().skip(max).next() {
                &s[..i]
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
            // If we're under the maximum width, check if we're over the minimum
            // width, if so it's as easy as just emitting the string.
            Some(width) if s.chars().count() >= width => {
                self.buf.write_str(s)
            }
            // If we're under both the maximum and the minimum width, then fill
            // up the minimum width with the specified string + some alignment.
            Some(width) => {
                let align = rt::v1::Alignment::Left;
                self.with_padding(width - s.chars().count(), align, |me| {
                    me.buf.write_str(s)
                })
            }
        }
    }

    /// Runs a callback, emitting the correct padding either before or
    /// afterwards depending on whether right or left alignment is requested.
    fn with_padding<F>(&mut self, padding: usize, default: rt::v1::Alignment,
                       f: F) -> Result
        where F: FnOnce(&mut Formatter) -> Result,
    {
        let align = match self.align {
            rt::v1::Alignment::Unknown => default,
            _ => self.align
        };

        let (pre_pad, post_pad) = match align {
            rt::v1::Alignment::Left => (0, padding),
            rt::v1::Alignment::Right |
            rt::v1::Alignment::Unknown => (padding, 0),
            rt::v1::Alignment::Center => (padding / 2, (padding + 1) / 2),
        };

        let mut fill = [0; 4];
        let fill = self.fill.encode_utf8(&mut fill);

        for _ in 0..pre_pad {
            self.buf.write_str(fill)?;
        }

        f(self)?;

        for _ in 0..post_pad {
            self.buf.write_str(fill)?;
        }

        Ok(())
    }

    /// Takes the formatted parts and applies the padding.
    /// Assumes that the caller already has rendered the parts with required precision,
    /// so that `self.precision` can be ignored.
    fn pad_formatted_parts(&mut self, formatted: &flt2dec::Formatted) -> Result {
        if let Some(mut width) = self.width {
            // for the sign-aware zero padding, we render the sign first and
            // behave as if we had no sign from the beginning.
            let mut formatted = formatted.clone();
            let mut align = self.align;
            let old_fill = self.fill;
            if self.sign_aware_zero_pad() {
                // a sign always goes first
                let sign = unsafe { str::from_utf8_unchecked(formatted.sign) };
                self.buf.write_str(sign)?;

                // remove the sign from the formatted parts
                formatted.sign = b"";
                width = if width < sign.len() { 0 } else { width - sign.len() };
                align = rt::v1::Alignment::Right;
                self.fill = '0';
            }

            // remaining parts go through the ordinary padding process.
            let len = formatted.len();
            let ret = if width <= len { // no padding
                self.write_formatted_parts(&formatted)
            } else {
                self.with_padding(width - len, align, |f| {
                    f.write_formatted_parts(&formatted)
                })
            };
            self.fill = old_fill;
            ret
        } else {
            // this is the common case and we take a shortcut
            self.write_formatted_parts(formatted)
        }
    }

    fn write_formatted_parts(&mut self, formatted: &flt2dec::Formatted) -> Result {
        fn write_bytes(buf: &mut Write, s: &[u8]) -> Result {
            buf.write_str(unsafe { str::from_utf8_unchecked(s) })
        }

        if !formatted.sign.is_empty() {
            write_bytes(self.buf, formatted.sign)?;
        }
        for part in formatted.parts {
            match *part {
                flt2dec::Part::Zero(mut nzeroes) => {
                    const ZEROES: &'static str = // 64 zeroes
                        "0000000000000000000000000000000000000000000000000000000000000000";
                    while nzeroes > ZEROES.len() {
                        self.buf.write_str(ZEROES)?;
                        nzeroes -= ZEROES.len();
                    }
                    if nzeroes > 0 {
                        self.buf.write_str(&ZEROES[..nzeroes])?;
                    }
                }
                flt2dec::Part::Num(mut v) => {
                    let mut s = [0; 5];
                    let len = part.len();
                    for c in s[..len].iter_mut().rev() {
                        *c = b'0' + (v % 10) as u8;
                        v /= 10;
                    }
                    write_bytes(self.buf, &s[..len])?;
                }
                flt2dec::Part::Copy(buf) => {
                    write_bytes(self.buf, buf)?;
                }
            }
        }
        Ok(())
    }

    /// Writes some data to the underlying buffer contained within this
    /// formatter.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn write_str(&mut self, data: &str) -> Result {
        self.buf.write_str(data)
    }

    /// Writes some formatted information into this instance
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn write_fmt(&mut self, fmt: Arguments) -> Result {
        write(self.buf, fmt)
    }

    /// Flags for formatting (packed version of rt::Flag)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn flags(&self) -> u32 { self.flags }

    /// Character used as 'fill' whenever there is alignment
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn fill(&self) -> char { self.fill }

    /// Flag indicating what form of alignment was requested
    #[unstable(feature = "fmt_flags_align", reason = "method was just created",
               issue = "27726")]
    pub fn align(&self) -> Alignment {
        match self.align {
            rt::v1::Alignment::Left => Alignment::Left,
            rt::v1::Alignment::Right => Alignment::Right,
            rt::v1::Alignment::Center => Alignment::Center,
            rt::v1::Alignment::Unknown => Alignment::Unknown,
        }
    }

    /// Optionally specified integer width that the output should be
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn width(&self) -> Option<usize> { self.width }

    /// Optionally specified precision for numeric types
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn precision(&self) -> Option<usize> { self.precision }

    /// Determines if the `+` flag was specified.
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn sign_plus(&self) -> bool { self.flags & (1 << FlagV1::SignPlus as u32) != 0 }

    /// Determines if the `-` flag was specified.
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn sign_minus(&self) -> bool { self.flags & (1 << FlagV1::SignMinus as u32) != 0 }

    /// Determines if the `#` flag was specified.
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn alternate(&self) -> bool { self.flags & (1 << FlagV1::Alternate as u32) != 0 }

    /// Determines if the `0` flag was specified.
    #[stable(feature = "fmt_flags", since = "1.5.0")]
    pub fn sign_aware_zero_pad(&self) -> bool {
        self.flags & (1 << FlagV1::SignAwareZeroPad as u32) != 0
    }

    /// Creates a `DebugStruct` builder designed to assist with creation of
    /// `fmt::Debug` implementations for structs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::fmt;
    ///
    /// struct Foo {
    ///     bar: i32,
    ///     baz: String,
    /// }
    ///
    /// impl fmt::Debug for Foo {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
    ///         fmt.debug_struct("Foo")
    ///             .field("bar", &self.bar)
    ///             .field("baz", &self.baz)
    ///             .finish()
    ///     }
    /// }
    ///
    /// // prints "Foo { bar: 10, baz: "Hello World" }"
    /// println!("{:?}", Foo { bar: 10, baz: "Hello World".to_string() });
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    #[inline]
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
    ///
    /// struct Foo(i32, String);
    ///
    /// impl fmt::Debug for Foo {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
    ///         fmt.debug_tuple("Foo")
    ///             .field(&self.0)
    ///             .field(&self.1)
    ///             .finish()
    ///     }
    /// }
    ///
    /// // prints "Foo(10, "Hello World")"
    /// println!("{:?}", Foo(10, "Hello World".to_string()));
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    #[inline]
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
    /// // prints "[10, 11]"
    /// println!("{:?}", Foo(vec![10, 11]));
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    #[inline]
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
    /// // prints "{10, 11}"
    /// println!("{:?}", Foo(vec![10, 11]));
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    #[inline]
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
    /// // prints "{"A": 10, "B": 11}"
    /// println!("{:?}", Foo(vec![("A".to_string(), 10), ("B".to_string(), 11)]));
    /// ```
    #[stable(feature = "debug_builders", since = "1.2.0")]
    #[inline]
    pub fn debug_map<'b>(&'b mut self) -> DebugMap<'b, 'a> {
        builders::debug_map_new(self)
    }
}

#[stable(since = "1.2.0", feature = "formatter_write")]
impl<'a> Write for Formatter<'a> {
    fn write_str(&mut self, s: &str) -> Result {
        self.buf.write_str(s)
    }

    fn write_char(&mut self, c: char) -> Result {
        self.buf.write_char(c)
    }

    fn write_fmt(&mut self, args: Arguments) -> Result {
        write(self.buf, args)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Display::fmt("an error occurred when formatting an argument", f)
    }
}

// Implementations of the core formatting traits

macro_rules! fmt_refs {
    ($($tr:ident),*) => {
        $(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T: ?Sized + $tr> $tr for &'a T {
            fn fmt(&self, f: &mut Formatter) -> Result { $tr::fmt(&**self, f) }
        }
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T: ?Sized + $tr> $tr for &'a mut T {
            fn fmt(&self, f: &mut Formatter) -> Result { $tr::fmt(&**self, f) }
        }
        )*
    }
}

fmt_refs! { Debug, Display, Octal, Binary, LowerHex, UpperHex, LowerExp, UpperExp }

#[unstable(feature = "never_type_impls", issue = "35121")]
impl Debug for ! {
    fn fmt(&self, _: &mut Formatter) -> Result {
        *self
    }
}

#[unstable(feature = "never_type_impls", issue = "35121")]
impl Display for ! {
    fn fmt(&self, _: &mut Formatter) -> Result {
        *self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for bool {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Display::fmt(self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Display for bool {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Display::fmt(if *self { "true" } else { "false" }, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for str {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.write_char('"')?;
        let mut from = 0;
        for (i, c) in self.char_indices() {
            let esc = c.escape_debug();
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
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.pad(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for char {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.write_char('\'')?;
        for c in self.escape_debug() {
            f.write_char(c)?
        }
        f.write_char('\'')
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Display for char {
    fn fmt(&self, f: &mut Formatter) -> Result {
        if f.width.is_none() && f.precision.is_none() {
            f.write_char(*self)
        } else {
            f.pad(self.encode_utf8(&mut [0; 4]))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Pointer for *const T {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let old_width = f.width;
        let old_flags = f.flags;

        // The alternate flag is already treated by LowerHex as being special-
        // it denotes whether to prefix with 0x. We use it to work out whether
        // or not to zero extend, and then unconditionally set it to get the
        // prefix.
        if f.alternate() {
            f.flags |= 1 << (FlagV1::SignAwareZeroPad as u32);

            if let None = f.width {
                f.width = Some(((mem::size_of::<usize>() * 8) / 4) + 2);
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
    fn fmt(&self, f: &mut Formatter) -> Result {
        Pointer::fmt(&(*self as *const T), f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized> Pointer for &'a T {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Pointer::fmt(&(*self as *const T), f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized> Pointer for &'a mut T {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Pointer::fmt(&(&**self as *const T), f)
    }
}

// Implementation of Display/Debug for various core types

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Debug for *const T {
    fn fmt(&self, f: &mut Formatter) -> Result { Pointer::fmt(self, f) }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Debug for *mut T {
    fn fmt(&self, f: &mut Formatter) -> Result { Pointer::fmt(self, f) }
}

macro_rules! peel {
    ($name:ident, $($other:ident,)*) => (tuple! { $($other,)* })
}

macro_rules! tuple {
    () => ();
    ( $($name:ident,)+ ) => (
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<$($name:Debug),*> Debug for ($($name,)*) {
            #[allow(non_snake_case, unused_assignments, deprecated)]
            fn fmt(&self, f: &mut Formatter) -> Result {
                let mut builder = f.debug_tuple("");
                let ($(ref $name,)*) = *self;
                $(
                    builder.field($name);
                )*

                builder.finish()
            }
        }
        peel! { $($name,)* }
    )
}

tuple! { T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, }

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Debug> Debug for [T] {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for () {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.pad("()")
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Debug for PhantomData<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.pad("PhantomData")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Copy + Debug> Debug for Cell<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.debug_struct("Cell")
            .field("value", &self.get())
            .finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Debug> Debug for RefCell<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self.try_borrow() {
            Ok(borrow) => {
                f.debug_struct("RefCell")
                    .field("value", &borrow)
                    .finish()
            }
            Err(_) => {
                f.debug_struct("RefCell")
                    .field("value", &"<borrowed>")
                    .finish()
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'b, T: ?Sized + Debug> Debug for Ref<'b, T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'b, T: ?Sized + Debug> Debug for RefMut<'b, T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Debug::fmt(&*(self.deref()), f)
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: ?Sized + Debug> Debug for UnsafeCell<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.pad("UnsafeCell")
    }
}

// If you expected tests to be here, look instead at the run-pass/ifmt.rs test,
// it's a lot easier than creating all of the rt::Piece structures here.
