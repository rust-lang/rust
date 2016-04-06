// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(primitive = "bool")]
//
/// The boolean type.
///
/// The `bool` represents a value, which could only be either `true` or `false`.
///
/// # Basic usage
///
/// `bool` implements various traits, such as [`BitAnd`], [`BitOr`], [`Not`], etc.,
/// which allow us to perform boolean operations using `&`, `|` and `!`.
///
/// [`if`] always demands a `bool` value. [`assert!`], being an important macro in testing,
/// checks whether an expression returns `true`.
///
/// ```
/// let bool_val = true & false | false;
/// assert!(!bool_val);
/// ```
///
/// [`assert!`]: macro.assert!.html
/// [`if` conditionals]: ../book/if.html
/// [`BitAnd`]: ops/trait.BitAnd.html
/// [`BitOr`]: ops/trait.BitOr.html
/// [`Not`]: ops/trait.Not.html
///
/// # Examples
///
/// A trivial example of the usage of `bool`,
///
/// ```
/// let praise_the_borrow_checker = true;
///
/// // using the `if` conditional
/// if praise_the_borrow_checker {
///     println!("oh, yeah!");
/// } else {
///     println!("what?!!");
/// }
///
/// // ... or, a match pattern
/// match praise_the_borrow_checker {
///     true => println!("keep praising!"),
///     false => println!("you should praise!"),
/// }
/// ```
///
/// Also, since `bool` implements the [`Copy`](marker/trait.Copy.html) trait, we don't
/// have to worry about the move semantics (just like the integer and float primitives).
mod prim_bool { }

#[doc(primitive = "char")]
//
/// A character type.
///
/// The `char` type represents a single character. More specifically, since
/// 'character' isn't a well-defined concept in Unicode, `char` is a '[Unicode
/// scalar value]', which is similar to, but not the same as, a '[Unicode code
/// point]'.
///
/// [Unicode scalar value]: http://www.unicode.org/glossary/#unicode_scalar_value
/// [Unicode code point]: http://www.unicode.org/glossary/#code_point
///
/// This documentation describes a number of methods and trait implementations on the
/// `char` type. For technical reasons, there is additional, separate
/// documentation in [the `std::char` module](char/index.html) as well.
///
/// # Representation
///
/// `char` is always four bytes in size. This is a different representation than
/// a given character would have as part of a [`String`]. For example:
///
/// ```
/// let v = vec!['h', 'e', 'l', 'l', 'o'];
///
/// // five elements times four bytes for each element
/// assert_eq!(20, v.len() * std::mem::size_of::<char>());
///
/// let s = String::from("hello");
///
/// // five elements times one byte per element
/// assert_eq!(5, s.len() * std::mem::size_of::<u8>());
/// ```
///
/// [`String`]: string/struct.String.html
///
/// As always, remember that a human intuition for 'character' may not map to
/// Unicode's definitions. For example, emoji symbols such as '❤️' can be more
/// than one Unicode code point; this ❤️ in particular is two:
///
/// ```
/// let s = String::from("❤️");
///
/// // we get two chars out of a single ❤️
/// let mut iter = s.chars();
/// assert_eq!(Some('\u{2764}'), iter.next());
/// assert_eq!(Some('\u{fe0f}'), iter.next());
/// assert_eq!(None, iter.next());
/// ```
///
/// This means it won't fit into a `char`. Trying to create a literal with
/// `let heart = '❤️';` gives an error:
///
/// ```text
/// error: character literal may only contain one codepoint: '❤
/// let heart = '❤️';
///             ^~
/// ```
///
/// Another implication of the 4-byte fixed size of a `char` is that
/// per-`char` processing can end up using a lot more memory:
///
/// ```
/// let s = String::from("love: ❤️");
/// let v: Vec<char> = s.chars().collect();
///
/// assert_eq!(12, s.len() * std::mem::size_of::<u8>());
/// assert_eq!(32, v.len() * std::mem::size_of::<char>());
/// ```
mod prim_char { }

#[doc(primitive = "unit")]
//
/// The `()` type, sometimes called "unit" or "nil".
///
/// The `()` type has exactly one value `()`, and is used when there
/// is no other meaningful value that could be returned. `()` is most
/// commonly seen implicitly: functions without a `-> ...` implicitly
/// have return type `()`, that is, these are equivalent:
///
/// ```rust
/// fn long() -> () {}
///
/// fn short() {}
/// ```
///
/// The semicolon `;` can be used to discard the result of an
/// expression at the end of a block, making the expression (and thus
/// the block) evaluate to `()`. For example,
///
/// ```rust
/// fn returns_i64() -> i64 {
///     1i64
/// }
/// fn returns_unit() {
///     1i64;
/// }
///
/// let is_i64 = {
///     returns_i64()
/// };
/// let is_unit = {
///     returns_i64();
/// };
/// ```
///
mod prim_unit { }

#[doc(primitive = "pointer")]
//
/// Raw, unsafe pointers, `*const T`, and `*mut T`.
///
/// Working with raw pointers in Rust is uncommon,
/// typically limited to a few patterns.
///
/// Use the `null` function to create null pointers, and the `is_null` method
/// of the `*const T` type  to check for null. The `*const T` type also defines
/// the `offset` method, for pointer math.
///
/// # Common ways to create raw pointers
///
/// ## 1. Coerce a reference (`&T`) or mutable reference (`&mut T`).
///
/// ```
/// let my_num: i32 = 10;
/// let my_num_ptr: *const i32 = &my_num;
/// let mut my_speed: i32 = 88;
/// let my_speed_ptr: *mut i32 = &mut my_speed;
/// ```
///
/// To get a pointer to a boxed value, dereference the box:
///
/// ```
/// let my_num: Box<i32> = Box::new(10);
/// let my_num_ptr: *const i32 = &*my_num;
/// let mut my_speed: Box<i32> = Box::new(88);
/// let my_speed_ptr: *mut i32 = &mut *my_speed;
/// ```
///
/// This does not take ownership of the original allocation
/// and requires no resource management later,
/// but you must not use the pointer after its lifetime.
///
/// ## 2. Consume a box (`Box<T>`).
///
/// The `into_raw` function consumes a box and returns
/// the raw pointer. It doesn't destroy `T` or deallocate any memory.
///
/// ```
/// let my_speed: Box<i32> = Box::new(88);
/// let my_speed: *mut i32 = Box::into_raw(my_speed);
///
/// // By taking ownership of the original `Box<T>` though
/// // we are obligated to put it together later to be destroyed.
/// unsafe {
///     drop(Box::from_raw(my_speed));
/// }
/// ```
///
/// Note that here the call to `drop` is for clarity - it indicates
/// that we are done with the given value and it should be destroyed.
///
/// ## 3. Get it from C.
///
/// ```
/// # #![feature(libc)]
/// extern crate libc;
///
/// use std::mem;
///
/// fn main() {
///     unsafe {
///         let my_num: *mut i32 = libc::malloc(mem::size_of::<i32>() as libc::size_t) as *mut i32;
///         if my_num.is_null() {
///             panic!("failed to allocate memory");
///         }
///         libc::free(my_num as *mut libc::c_void);
///     }
/// }
/// ```
///
/// Usually you wouldn't literally use `malloc` and `free` from Rust,
/// but C APIs hand out a lot of pointers generally, so are a common source
/// of raw pointers in Rust.
///
/// *[See also the `std::ptr` module](ptr/index.html).*
///
mod prim_pointer { }

#[doc(primitive = "array")]
//
/// A fixed-size array, denoted `[T; N]`, for the element type, `T`, and the
/// non-negative compile time constant size, `N`.
///
/// Arrays values are created either with an explicit expression that lists
/// each element: `[x, y, z]` or a repeat expression: `[x; N]`. The repeat
/// expression requires that the element type is `Copy`.
///
/// The type `[T; N]` is `Copy` if `T: Copy`.
///
/// Arrays of sizes from 0 to 32 (inclusive) implement the following traits if
/// the element type allows it:
///
/// - `Clone` (only if `T: Copy`)
/// - `Debug`
/// - `IntoIterator` (implemented for `&[T; N]` and `&mut [T; N]`)
/// - `PartialEq`, `PartialOrd`, `Ord`, `Eq`
/// - `Hash`
/// - `AsRef`, `AsMut`
/// - `Borrow`, `BorrowMut`
/// - `Default`
///
/// Arrays coerce to [slices (`[T]`)][slice], so their methods can be called on
/// arrays.
///
/// [slice]: primitive.slice.html
///
/// Rust does not currently support generics over the size of an array type.
///
/// # Examples
///
/// ```
/// let mut array: [i32; 3] = [0; 3];
///
/// array[1] = 1;
/// array[2] = 2;
///
/// assert_eq!([1, 2], &array[1..]);
///
/// // This loop prints: 0 1 2
/// for x in &array {
///     print!("{} ", x);
/// }
///
/// ```
///
mod prim_array { }

#[doc(primitive = "slice")]
//
/// A dynamically-sized view into a contiguous sequence, `[T]`.
///
/// Slices are a view into a block of memory represented as a pointer and a
/// length.
///
/// ```
/// // slicing a Vec
/// let vec = vec![1, 2, 3];
/// let int_slice = &vec[..];
/// // coercing an array to a slice
/// let str_slice: &[&str] = &["one", "two", "three"];
/// ```
///
/// Slices are either mutable or shared. The shared slice type is `&[T]`,
/// while the mutable slice type is `&mut [T]`, where `T` represents the element
/// type. For example, you can mutate the block of memory that a mutable slice
/// points to:
///
/// ```
/// let x = &mut [1, 2, 3];
/// x[1] = 7;
/// assert_eq!(x, &[1, 7, 3]);
/// ```
///
/// *[See also the `std::slice` module](slice/index.html).*
///
mod prim_slice { }

#[doc(primitive = "str")]
//
/// String slices.
///
/// The `str` type, also called a 'string slice', is the most primitive string
/// type. It is usually seen in its borrowed form, `&str`. It is also the type
/// of string literals, `&'static str`.
///
/// Strings slices are always valid UTF-8.
///
/// This documentation describes a number of methods and trait implementations
/// on the `str` type. For technical reasons, there is additional, separate
/// documentation in [the `std::str` module](str/index.html) as well.
///
/// # Examples
///
/// String literals are string slices:
///
/// ```
/// let hello = "Hello, world!";
///
/// // with an explicit type annotation
/// let hello: &'static str = "Hello, world!";
/// ```
///
/// They are `'static` because they're stored directly in the final binary, and
/// so will be valid for the `'static` duration.
///
/// # Representation
///
/// A `&str` is made up of two components: a pointer to some bytes, and a
/// length. You can look at these with the [`.as_ptr()`] and [`len()`] methods:
///
/// ```
/// use std::slice;
/// use std::str;
///
/// let story = "Once upon a time...";
///
/// let ptr = story.as_ptr();
/// let len = story.len();
///
/// // story has nineteen bytes
/// assert_eq!(19, len);
///
/// // We can re-build a str out of ptr and len. This is all unsafe because
/// // we are responsible for making sure the two components are valid:
/// let s = unsafe {
///     // First, we build a &[u8]...
///     let slice = slice::from_raw_parts(ptr, len);
///
///     // ... and then convert that slice into a string slice
///     str::from_utf8(slice)
/// };
///
/// assert_eq!(s, Ok(story));
/// ```
///
/// [`.as_ptr()`]: #method.as_ptr
/// [`len()`]: #method.len
mod prim_str { }

#[doc(primitive = "tuple")]
//
/// A finite heterogeneous sequence, `(T, U, ..)`.
///
/// Let's cover each of those in turn:
///
/// Tuples are *finite*. In other words, a tuple has a length. Here's a tuple
/// of length `3`:
///
/// ```
/// ("hello", 5, 'c');
/// ```
///
/// 'Length' is also sometimes called 'arity' here; each tuple of a different
/// length is a different, distinct type.
///
/// Tuples are *heterogeneous*. This means that each element of the tuple can
/// have a different type. In that tuple above, it has the type:
///
/// ```rust,ignore
/// (&'static str, i32, char)
/// ```
///
/// Tuples are a *sequence*. This means that they can be accessed by position;
/// this is called 'tuple indexing', and it looks like this:
///
/// ```rust
/// let tuple = ("hello", 5, 'c');
///
/// assert_eq!(tuple.0, "hello");
/// assert_eq!(tuple.1, 5);
/// assert_eq!(tuple.2, 'c');
/// ```
///
/// For more about tuples, see [the book](../book/primitive-types.html#tuples).
///
/// # Trait implementations
///
/// If every type inside a tuple implements one of the following traits, then a
/// tuple itself also implements it.
///
/// * [`Clone`]
/// * [`Copy`]
/// * [`PartialEq`]
/// * [`Eq`]
/// * [`PartialOrd`]
/// * [`Ord`]
/// * [`Debug`]
/// * [`Default`]
/// * [`Hash`]
///
/// [`Clone`]: clone/trait.Clone.html
/// [`Copy`]: marker/trait.Copy.html
/// [`PartialEq`]: cmp/trait.PartialEq.html
/// [`Eq`]: cmp/trait.Eq.html
/// [`PartialOrd`]: cmp/trait.PartialOrd.html
/// [`Ord`]: cmp/trait.Ord.html
/// [`Debug`]: fmt/trait.Debug.html
/// [`Default`]: default/trait.Default.html
/// [`Hash`]: hash/trait.Hash.html
///
/// Due to a temporary restriction in Rust's type system, these traits are only
/// implemented on tuples of arity 32 or less. In the future, this may change.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let tuple = ("hello", 5, 'c');
///
/// assert_eq!(tuple.0, "hello");
/// ```
///
/// Tuples are often used as a return type when you want to return more than
/// one value:
///
/// ```
/// fn calculate_point() -> (i32, i32) {
///     // Don't do a calculation, that's not the point of the example
///     (4, 5)
/// }
///
/// let point = calculate_point();
///
/// assert_eq!(point.0, 4);
/// assert_eq!(point.1, 5);
///
/// // Combining this with patterns can be nicer.
///
/// let (x, y) = calculate_point();
///
/// assert_eq!(x, 4);
/// assert_eq!(y, 5);
/// ```
///
mod prim_tuple { }

#[doc(primitive = "f32")]
/// The 32-bit floating point type.
///
/// *[See also the `std::f32` module](f32/index.html).*
///
/// However, please note that examples are shared between the `f64` and `f32`
/// primitive types. So it's normal if you see usage of `f64` in there.
///
mod prim_f32 { }

#[doc(primitive = "f64")]
//
/// The 64-bit floating point type.
///
/// *[See also the `std::f64` module](f64/index.html).*
///
/// However, please note that examples are shared between the `f64` and `f32`
/// primitive types. So it's normal if you see usage of `f32` in there.
///
mod prim_f64 { }

#[doc(primitive = "i8")]
//
/// The 8-bit signed integer type.
///
/// *[See also the `std::i8` module](i8/index.html).*
///
mod prim_i8 { }

#[doc(primitive = "i16")]
//
/// The 16-bit signed integer type.
///
/// *[See also the `std::i16` module](i16/index.html).*
///
mod prim_i16 { }

#[doc(primitive = "i32")]
//
/// The 32-bit signed integer type.
///
/// *[See also the `std::i32` module](i32/index.html).*
///
mod prim_i32 { }

#[doc(primitive = "i64")]
//
/// The 64-bit signed integer type.
///
/// *[See also the `std::i64` module](i64/index.html).*
///
mod prim_i64 { }

#[doc(primitive = "u8")]
//
/// The 8-bit unsigned integer type.
///
/// *[See also the `std::u8` module](u8/index.html).*
///
mod prim_u8 { }

#[doc(primitive = "u16")]
//
/// The 16-bit unsigned integer type.
///
/// *[See also the `std::u16` module](u16/index.html).*
///
mod prim_u16 { }

#[doc(primitive = "u32")]
//
/// The 32-bit unsigned integer type.
///
/// *[See also the `std::u32` module](u32/index.html).*
///
mod prim_u32 { }

#[doc(primitive = "u64")]
//
/// The 64-bit unsigned integer type.
///
/// *[See also the `std::u64` module](u64/index.html).*
///
mod prim_u64 { }

#[doc(primitive = "isize")]
//
/// The pointer-sized signed integer type.
///
/// *[See also the `std::isize` module](isize/index.html).*
///
mod prim_isize { }

#[doc(primitive = "usize")]
//
/// The pointer-sized unsigned integer type.
///
/// *[See also the `std::usize` module](usize/index.html).*
///
mod prim_usize { }
