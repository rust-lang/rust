//! A UTF-8–encoded, growable string.
//!
//! This module contains the [`String`] type, the [`ToString`] trait for
//! converting to strings, and several error types that may result from
//! working with [`String`]s.
//!
//! # Examples
//!
//! There are multiple ways to create a new [`String`] from a string literal:
//!
//! ```
//! let s = "Hello".to_string();
//!
//! let s = String::from("world");
//! let s: String = "also this".into();
//! ```
//!
//! You can create a new [`String`] from an existing one by concatenating with
//! `+`:
//!
//! ```
//! let s = "Hello".to_string();
//!
//! let message = s + " world!";
//! ```
//!
//! If you have a vector of valid UTF-8 bytes, you can make a [`String`] out of
//! it. You can do the reverse too.
//!
//! ```
//! let sparkle_heart = vec![240, 159, 146, 150];
//!
//! // We know these bytes are valid, so we'll use `unwrap()`.
//! let sparkle_heart = String::from_utf8(sparkle_heart).unwrap();
//!
//! assert_eq!("💖", sparkle_heart);
//!
//! let bytes = sparkle_heart.into_bytes();
//!
//! assert_eq!(bytes, [240, 159, 146, 150]);
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

use core::fmt;

use crate::alloc::{Allocator, Global};
#[cfg(not(no_global_oom_handling))]
use crate::borrow::{Cow, ToOwned};

pub mod string;

/// A UTF-8–encoded, growable string.
///
/// The `String` type is the most common string type that has ownership over the
/// contents of the string. It has a close relationship with its borrowed
/// counterpart, the primitive [`str`].
///
/// # Examples
///
/// You can create a `String` from [a literal string][`&str`] with [`String::from`]:
///
/// [`String::from`]: From::from
///
/// ```
/// let hello = String::from("Hello, world!");
/// ```
///
/// You can append a [`char`] to a `String` with the [`push`] method, and
/// append a [`&str`] with the [`push_str`] method:
///
/// ```
/// let mut hello = String::from("Hello, ");
///
/// hello.push('w');
/// hello.push_str("orld!");
/// ```
///
/// [`push`]: String::push
/// [`push_str`]: String::push_str
///
/// If you have a vector of UTF-8 bytes, you can create a `String` from it with
/// the [`from_utf8`] method:
///
/// ```
/// // some bytes, in a vector
/// let sparkle_heart = vec![240, 159, 146, 150];
///
/// // We know these bytes are valid, so we'll use `unwrap()`.
/// let sparkle_heart = String::from_utf8(sparkle_heart).unwrap();
///
/// assert_eq!("💖", sparkle_heart);
/// ```
///
/// [`from_utf8`]: String::from_utf8
///
/// # UTF-8
///
/// `String`s are always valid UTF-8. If you need a non-UTF-8 string, consider
/// [`OsString`]. It is similar, but without the UTF-8 constraint. Because UTF-8
/// is a variable width encoding, `String`s are typically smaller than an array of
/// the same `chars`:
///
/// ```
/// use std::mem;
///
/// // `s` is ASCII which represents each `char` as one byte
/// let s = "hello";
/// assert_eq!(s.len(), 5);
///
/// // A `char` array with the same contents would be longer because
/// // every `char` is four bytes
/// let s = ['h', 'e', 'l', 'l', 'o'];
/// let size: usize = s.into_iter().map(|c| mem::size_of_val(&c)).sum();
/// assert_eq!(size, 20);
///
/// // However, for non-ASCII strings, the difference will be smaller
/// // and sometimes they are the same
/// let s = "💖💖💖💖💖";
/// assert_eq!(s.len(), 20);
///
/// let s = ['💖', '💖', '💖', '💖', '💖'];
/// let size: usize = s.into_iter().map(|c| mem::size_of_val(&c)).sum();
/// assert_eq!(size, 20);
/// ```
///
/// This raises interesting questions as to how `s[i]` should work.
/// What should `i` be here? Several options include byte indices and
/// `char` indices but, because of UTF-8 encoding, only byte indices
/// would provide constant time indexing. Getting the `i`th `char`, for
/// example, is available using [`chars`]:
///
/// ```
/// let s = "hello";
/// let third_character = s.chars().nth(2);
/// assert_eq!(third_character, Some('l'));
///
/// let s = "💖💖💖💖💖";
/// let third_character = s.chars().nth(2);
/// assert_eq!(third_character, Some('💖'));
/// ```
///
/// Next, what should `s[i]` return? Because indexing returns a reference
/// to underlying data it could be `&u8`, `&[u8]`, or something else similar.
/// Since we're only providing one index, `&u8` makes the most sense but that
/// might not be what the user expects and can be explicitly achieved with
/// [`as_bytes()`]:
///
/// ```
/// // The first byte is 104 - the byte value of `'h'`
/// let s = "hello";
/// assert_eq!(s.as_bytes()[0], 104);
/// // or
/// assert_eq!(s.as_bytes()[0], b'h');
///
/// // The first byte is 240 which isn't obviously useful
/// let s = "💖💖💖💖💖";
/// assert_eq!(s.as_bytes()[0], 240);
/// ```
///
/// Due to these ambiguities/restrictions, indexing with a `usize` is simply
/// forbidden:
///
/// ```compile_fail,E0277
/// let s = "hello";
///
/// // The following will not compile!
/// println!("The first letter of s is {}", s[0]);
/// ```
///
/// It is more clear, however, how `&s[i..j]` should work (that is,
/// indexing with a range). It should accept byte indices (to be constant-time)
/// and return a `&str` which is UTF-8 encoded. This is also called "string slicing".
/// Note this will panic if the byte indices provided are not character
/// boundaries - see [`is_char_boundary`] for more details. See the implementations
/// for [`SliceIndex<str>`] for more details on string slicing. For a non-panicking
/// version of string slicing, see [`get`].
///
/// [`OsString`]: ../../std/ffi/struct.OsString.html "ffi::OsString"
/// [`SliceIndex<str>`]: core::slice::SliceIndex
/// [`as_bytes()`]: str::as_bytes
/// [`get`]: str::get
/// [`is_char_boundary`]: str::is_char_boundary
///
/// The [`bytes`] and [`chars`] methods return iterators over the bytes and
/// codepoints of the string, respectively. To iterate over codepoints along
/// with byte indices, use [`char_indices`].
///
/// [`bytes`]: str::bytes
/// [`chars`]: str::chars
/// [`char_indices`]: str::char_indices
///
/// # Deref
///
/// `String` implements <code>[Deref]<Target = [str]></code>, and so inherits all of [`str`]'s
/// methods. In addition, this means that you can pass a `String` to a
/// function which takes a [`&str`] by using an ampersand (`&`):
///
/// ```
/// fn takes_str(s: &str) { }
///
/// let s = String::from("Hello");
///
/// takes_str(&s);
/// ```
///
/// This will create a [`&str`] from the `String` and pass it in. This
/// conversion is very inexpensive, and so generally, functions will accept
/// [`&str`]s as arguments unless they need a `String` for some specific
/// reason.
///
/// In certain cases Rust doesn't have enough information to make this
/// conversion, known as [`Deref`] coercion. In the following example a string
/// slice [`&'a str`][`&str`] implements the trait `TraitExample`, and the function
/// `example_func` takes anything that implements the trait. In this case Rust
/// would need to make two implicit conversions, which Rust doesn't have the
/// means to do. For that reason, the following example will not compile.
///
/// ```compile_fail,E0277
/// trait TraitExample {}
///
/// impl<'a> TraitExample for &'a str {}
///
/// fn example_func<A: TraitExample>(example_arg: A) {}
///
/// let example_string = String::from("example_string");
/// example_func(&example_string);
/// ```
///
/// There are two options that would work instead. The first would be to
/// change the line `example_func(&example_string);` to
/// `example_func(example_string.as_str());`, using the method [`as_str()`]
/// to explicitly extract the string slice containing the string. The second
/// way changes `example_func(&example_string);` to
/// `example_func(&*example_string);`. In this case we are dereferencing a
/// `String` to a [`str`], then referencing the [`str`] back to
/// [`&str`]. The second way is more idiomatic, however both work to do the
/// conversion explicitly rather than relying on the implicit conversion.
///
/// # Representation
///
/// A `String` is made up of three components: a pointer to some bytes, a
/// length, and a capacity. The pointer points to an internal buffer `String`
/// uses to store its data. The length is the number of bytes currently stored
/// in the buffer, and the capacity is the size of the buffer in bytes. As such,
/// the length will always be less than or equal to the capacity.
///
/// This buffer is always stored on the heap.
///
/// You can look at these with the [`as_ptr`], [`len`], and [`capacity`]
/// methods:
///
/// ```
/// use std::mem;
///
/// let story = String::from("Once upon a time...");
///
// FIXME Update this when vec_into_raw_parts is stabilized
/// // Prevent automatically dropping the String's data
/// let mut story = mem::ManuallyDrop::new(story);
///
/// let ptr = story.as_mut_ptr();
/// let len = story.len();
/// let capacity = story.capacity();
///
/// // story has nineteen bytes
/// assert_eq!(19, len);
///
/// // We can re-build a String out of ptr, len, and capacity. This is all
/// // unsafe because we are responsible for making sure the components are
/// // valid:
/// let s = unsafe { String::from_raw_parts(ptr, len, capacity) } ;
///
/// assert_eq!(String::from("Once upon a time..."), s);
/// ```
///
/// [`as_ptr`]: str::as_ptr
/// [`len`]: String::len
/// [`capacity`]: String::capacity
///
/// If a `String` has enough capacity, adding elements to it will not
/// re-allocate. For example, consider this program:
///
/// ```
/// let mut s = String::new();
///
/// println!("{}", s.capacity());
///
/// for _ in 0..5 {
///     s.push_str("hello");
///     println!("{}", s.capacity());
/// }
/// ```
///
/// This will output the following:
///
/// ```text
/// 0
/// 8
/// 16
/// 16
/// 32
/// 32
/// ```
///
///
/// At first, we have no memory allocated at all, but as we append to the
/// string, it increases its capacity appropriately. If we instead use the
/// [`with_capacity`] method to allocate the correct capacity initially:
///
/// ```
/// let mut s = String::with_capacity(25);
///
/// println!("{}", s.capacity());
///
/// for _ in 0..5 {
///     s.push_str("hello");
///     println!("{}", s.capacity());
/// }
/// ```
///
/// [`with_capacity`]: String::with_capacity
///
/// We end up with a different output:
///
/// ```text
/// 25
/// 25
/// 25
/// 25
/// 25
/// 25
/// ```
///
/// Here, there's no need to allocate more memory inside the loop.
///
/// [str]: prim@str "str"
/// [`str`]: prim@str "str"
/// [`&str`]: prim@str "&str"
/// [Deref]: core::ops::Deref "ops::Deref"
/// [`Deref`]: core::ops::Deref "ops::Deref"
/// [`as_str()`]: String::as_str
#[stable(feature = "rust1", since = "1.0.0")]
pub type String = string::String<Global>;

/// A possible error value when converting a `String` from a UTF-8 byte vector.
///
/// This type is the error type for the [`from_utf8`] method on [`String`]. It
/// is designed in such a way to carefully avoid reallocations: the
/// [`into_bytes`] method will give back the byte vector that was used in the
/// conversion attempt.
///
/// [`from_utf8`]: String::from_utf8
/// [`into_bytes`]: FromUtf8Error::into_bytes
///
/// The [`Utf8Error`] type provided by [`std::str`] represents an error that may
/// occur when converting a slice of [`u8`]s to a [`&str`]. In this sense, it's
/// an analogue to `FromUtf8Error`, and you can get one from a `FromUtf8Error`
/// through the [`utf8_error`] method.
///
/// [`Utf8Error`]: core::str::Utf8Error "std::str::Utf8Error"
/// [`std::str`]: core::str "std::str"
/// [`&str`]: prim@str "&str"
/// [`utf8_error`]: FromUtf8Error::utf8_error
///
/// # Examples
///
/// ```
/// // some invalid bytes, in a vector
/// let bytes = vec![0, 159];
///
/// let value = String::from_utf8(bytes);
///
/// assert!(value.is_err());
/// assert_eq!(vec![0, 159], value.unwrap_err().into_bytes());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub type FromUtf8Error = string::FromUtf8Error<Global>;

#[doc(inline)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use string::FromUtf16Error;

/// A type alias for [`Infallible`].
///
/// This alias exists for backwards compatibility, and may be eventually deprecated.
///
/// [`Infallible`]: core::convert::Infallible "convert::Infallible"
#[stable(feature = "str_parse_error", since = "1.5.0")]
pub type ParseError = core::convert::Infallible;

/// A trait for converting a value to a `String`.
///
/// This trait is automatically implemented for any type which implements the
/// [`Display`] trait. As such, `ToString` shouldn't be implemented directly:
/// [`Display`] should be implemented instead, and you get the `ToString`
/// implementation for free.
///
/// [`Display`]: fmt::Display
#[cfg_attr(not(test), rustc_diagnostic_item = "ToString")]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ToString {
    /// Converts the given value to a `String`.
    ///
    /// # Examples
    ///
    /// ```
    /// let i = 5;
    /// let five = String::from("5");
    ///
    /// assert_eq!(five, i.to_string());
    /// ```
    #[rustc_conversion_suggestion]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "to_string_method")]
    fn to_string(&self) -> String;
}

/// # Panics
///
/// In this implementation, the `to_string` method panics
/// if the `Display` implementation returns an error.
/// This indicates an incorrect `Display` implementation
/// since `fmt::Write for String` never returns an error itself.
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Display + ?Sized> ToString for T {
    // A common guideline is to not inline generic functions. However,
    // removing `#[inline]` from this method causes non-negligible regressions.
    // See <https://github.com/rust-lang/rust/pull/74852>, the last attempt
    // to try to remove it.
    #[inline]
    default fn to_string(&self) -> String {
        let mut buf = String::new();
        let mut formatter = core::fmt::Formatter::new(&mut buf);
        // Bypass format_args!() to avoid write_str with zero-length strs
        fmt::Display::fmt(self, &mut formatter)
            .expect("a Display implementation returned an error unexpectedly");
        buf
    }
}

#[doc(hidden)]
#[cfg(not(no_global_oom_handling))]
#[unstable(feature = "ascii_char", issue = "110998")]
impl ToString for core::ascii::Char {
    #[inline]
    fn to_string(&self) -> String {
        self.as_str().to_owned()
    }
}

#[doc(hidden)]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "char_to_string_specialization", since = "1.46.0")]
impl ToString for char {
    #[inline]
    fn to_string(&self) -> String {
        String::from(self.encode_utf8(&mut [0; 4]))
    }
}

#[doc(hidden)]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "bool_to_string_specialization", since = "1.68.0")]
impl ToString for bool {
    #[inline]
    fn to_string(&self) -> String {
        String::from(if *self { "true" } else { "false" })
    }
}

#[doc(hidden)]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "u8_to_string_specialization", since = "1.54.0")]
impl ToString for u8 {
    #[inline]
    fn to_string(&self) -> String {
        let mut buf = String::with_capacity(3);
        let mut n = *self;
        if n >= 10 {
            if n >= 100 {
                buf.push((b'0' + n / 100) as char);
                n %= 100;
            }
            buf.push((b'0' + n / 10) as char);
            n %= 10;
        }
        buf.push((b'0' + n) as char);
        buf
    }
}

#[doc(hidden)]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "i8_to_string_specialization", since = "1.54.0")]
impl ToString for i8 {
    #[inline]
    fn to_string(&self) -> String {
        let mut buf = String::with_capacity(4);
        if self.is_negative() {
            buf.push('-');
        }
        let mut n = self.unsigned_abs();
        if n >= 10 {
            if n >= 100 {
                buf.push('1');
                n -= 100;
            }
            buf.push((b'0' + n / 10) as char);
            n %= 10;
        }
        buf.push((b'0' + n) as char);
        buf
    }
}

#[doc(hidden)]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "str_to_string_specialization", since = "1.9.0")]
impl ToString for str {
    #[inline]
    fn to_string(&self) -> String {
        String::from(self)
    }
}

#[doc(hidden)]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "cow_str_to_string_specialization", since = "1.17.0")]
impl ToString for Cow<'_, str> {
    #[inline]
    fn to_string(&self) -> String {
        self[..].to_owned()
    }
}

#[doc(hidden)]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "string_to_string_specialization", since = "1.17.0")]
impl<A: Allocator> ToString for string::String<A> {
    #[inline]
    fn to_string(&self) -> String {
        self[..].to_owned()
    }
}

#[doc(hidden)]
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "fmt_arguments_to_string_specialization", since = "1.71.0")]
impl ToString for fmt::Arguments<'_> {
    #[inline]
    fn to_string(&self) -> String {
        crate::fmt::format(*self)
    }
}

/// A draining iterator for `String`.
///
/// This struct is created by the [`drain`] method on [`String`]. See its
/// documentation for more.
///
/// [`drain`]: String::drain
#[stable(feature = "drain", since = "1.6.0")]
pub type Drain<'a> = string::Drain<'a, Global>;
