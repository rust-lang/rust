// `library/{std,core}/src/primitive_docs.rs` should have the same contents.
// These are different files so that relative links work properly without
// having to have `CARGO_PKG_NAME` set, but conceptually they should always be the same.
#[doc(primitive = "bool")]
#[doc(alias = "true")]
#[doc(alias = "false")]
/// The boolean type.
///
/// The `bool` represents a value, which could only be either [`true`] or [`false`]. If you cast
/// a `bool` into an integer, [`true`] will be 1 and [`false`] will be 0.
///
/// # Basic usage
///
/// `bool` implements various traits, such as [`BitAnd`], [`BitOr`], [`Not`], etc.,
/// which allow us to perform boolean operations using `&`, `|` and `!`.
///
/// [`if`] requires a `bool` value as its conditional. [`assert!`], which is an
/// important macro in testing, checks whether an expression is [`true`] and panics
/// if it isn't.
///
/// ```
/// let bool_val = true & false | false;
/// assert!(!bool_val);
/// ```
///
/// [`true`]: ../std/keyword.true.html
/// [`false`]: ../std/keyword.false.html
/// [`BitAnd`]: ops::BitAnd
/// [`BitOr`]: ops::BitOr
/// [`Not`]: ops::Not
/// [`if`]: ../std/keyword.if.html
///
/// # Examples
///
/// A trivial example of the usage of `bool`:
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
/// Also, since `bool` implements the [`Copy`] trait, we don't
/// have to worry about the move semantics (just like the integer and float primitives).
///
/// Now an example of `bool` cast to integer type:
///
/// ```
/// assert_eq!(true as i32, 1);
/// assert_eq!(false as i32, 0);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_bool {}

#[doc(primitive = "never")]
#[doc(alias = "!")]
//
/// The `!` type, also called "never".
///
/// `!` represents the type of computations which never resolve to any value at all. For example,
/// the [`exit`] function `fn exit(code: i32) -> !` exits the process without ever returning, and
/// so returns `!`.
///
/// `break`, `continue` and `return` expressions also have type `!`. For example we are allowed to
/// write:
///
/// ```
/// #![feature(never_type)]
/// # fn foo() -> u32 {
/// let x: ! = {
///     return 123
/// };
/// # }
/// ```
///
/// Although the `let` is pointless here, it illustrates the meaning of `!`. Since `x` is never
/// assigned a value (because `return` returns from the entire function), `x` can be given type
/// `!`. We could also replace `return 123` with a `panic!` or a never-ending `loop` and this code
/// would still be valid.
///
/// A more realistic usage of `!` is in this code:
///
/// ```
/// # fn get_a_number() -> Option<u32> { None }
/// # loop {
/// let num: u32 = match get_a_number() {
///     Some(num) => num,
///     None => break,
/// };
/// # }
/// ```
///
/// Both match arms must produce values of type [`u32`], but since `break` never produces a value
/// at all we know it can never produce a value which isn't a [`u32`]. This illustrates another
/// behaviour of the `!` type - expressions with type `!` will coerce into any other type.
///
/// [`u32`]: prim@u32
#[doc = concat!("[`exit`]: ", include_str!("../primitive_docs/process_exit.md"))]
///
/// # `!` and generics
///
/// ## Infallible errors
///
/// The main place you'll see `!` used explicitly is in generic code. Consider the [`FromStr`]
/// trait:
///
/// ```
/// trait FromStr: Sized {
///     type Err;
///     fn from_str(s: &str) -> Result<Self, Self::Err>;
/// }
/// ```
///
/// When implementing this trait for [`String`] we need to pick a type for [`Err`]. And since
/// converting a string into a string will never result in an error, the appropriate type is `!`.
/// (Currently the type actually used is an enum with no variants, though this is only because `!`
/// was added to Rust at a later date and it may change in the future.) With an [`Err`] type of
/// `!`, if we have to call [`String::from_str`] for some reason the result will be a
/// [`Result<String, !>`] which we can unpack like this:
///
/// ```
/// #![feature(exhaustive_patterns)]
/// use std::str::FromStr;
/// let Ok(s) = String::from_str("hello");
/// ```
///
/// Since the [`Err`] variant contains a `!`, it can never occur. If the `exhaustive_patterns`
/// feature is present this means we can exhaustively match on [`Result<T, !>`] by just taking the
/// [`Ok`] variant. This illustrates another behaviour of `!` - it can be used to "delete" certain
/// enum variants from generic types like `Result`.
///
/// ## Infinite loops
///
/// While [`Result<T, !>`] is very useful for removing errors, `!` can also be used to remove
/// successes as well. If we think of [`Result<T, !>`] as "if this function returns, it has not
/// errored," we get a very intuitive idea of [`Result<!, E>`] as well: if the function returns, it
/// *has* errored.
///
/// For example, consider the case of a simple web server, which can be simplified to:
///
/// ```ignore (hypothetical-example)
/// loop {
///     let (client, request) = get_request().expect("disconnected");
///     let response = request.process();
///     response.send(client);
/// }
/// ```
///
/// Currently, this isn't ideal, because we simply panic whenever we fail to get a new connection.
/// Instead, we'd like to keep track of this error, like this:
///
/// ```ignore (hypothetical-example)
/// loop {
///     match get_request() {
///         Err(err) => break err,
///         Ok((client, request)) => {
///             let response = request.process();
///             response.send(client);
///         },
///     }
/// }
/// ```
///
/// Now, when the server disconnects, we exit the loop with an error instead of panicking. While it
/// might be intuitive to simply return the error, we might want to wrap it in a [`Result<!, E>`]
/// instead:
///
/// ```ignore (hypothetical-example)
/// fn server_loop() -> Result<!, ConnectionError> {
///     loop {
///         let (client, request) = get_request()?;
///         let response = request.process();
///         response.send(client);
///     }
/// }
/// ```
///
/// Now, we can use `?` instead of `match`, and the return type makes a lot more sense: if the loop
/// ever stops, it means that an error occurred. We don't even have to wrap the loop in an `Ok`
/// because `!` coerces to `Result<!, ConnectionError>` automatically.
///
/// [`String::from_str`]: str::FromStr::from_str
#[doc = concat!("[`String`]: ", include_str!("../primitive_docs/string_string.md"))]
/// [`FromStr`]: str::FromStr
///
/// # `!` and traits
///
/// When writing your own traits, `!` should have an `impl` whenever there is an obvious `impl`
/// which doesn't `panic!`. The reason is that functions returning an `impl Trait` where `!`
/// does not have an `impl` of `Trait` cannot diverge as their only possible code path. In other
/// words, they can't return `!` from every code path. As an example, this code doesn't compile:
///
/// ```compile_fail
/// use std::ops::Add;
///
/// fn foo() -> impl Add<u32> {
///     unimplemented!()
/// }
/// ```
///
/// But this code does:
///
/// ```
/// use std::ops::Add;
///
/// fn foo() -> impl Add<u32> {
///     if true {
///         unimplemented!()
///     } else {
///         0
///     }
/// }
/// ```
///
/// The reason is that, in the first example, there are many possible types that `!` could coerce
/// to, because many types implement `Add<u32>`. However, in the second example,
/// the `else` branch returns a `0`, which the compiler infers from the return type to be of type
/// `u32`. Since `u32` is a concrete type, `!` can and will be coerced to it. See issue [#36375]
/// for more information on this quirk of `!`.
///
/// [#36375]: https://github.com/rust-lang/rust/issues/36375
///
/// As it turns out, though, most traits can have an `impl` for `!`. Take [`Debug`]
/// for example:
///
/// ```
/// #![feature(never_type)]
/// # use std::fmt;
/// # trait Debug {
/// #     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result;
/// # }
/// impl Debug for ! {
///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
///         *self
///     }
/// }
/// ```
///
/// Once again we're using `!`'s ability to coerce into any other type, in this case
/// [`fmt::Result`]. Since this method takes a `&!` as an argument we know that it can never be
/// called (because there is no value of type `!` for it to be called with). Writing `*self`
/// essentially tells the compiler "We know that this code can never be run, so just treat the
/// entire function body as having type [`fmt::Result`]". This pattern can be used a lot when
/// implementing traits for `!`. Generally, any trait which only has methods which take a `self`
/// parameter should have such an impl.
///
/// On the other hand, one trait which would not be appropriate to implement is [`Default`]:
///
/// ```
/// trait Default {
///     fn default() -> Self;
/// }
/// ```
///
/// Since `!` has no values, it has no default value either. It's true that we could write an
/// `impl` for this which simply panics, but the same is true for any type (we could `impl
/// Default` for (eg.) [`File`] by just making [`default()`] panic.)
///
#[doc = concat!("[`File`]: ", include_str!("../primitive_docs/fs_file.md"))]
/// [`Debug`]: fmt::Debug
/// [`default()`]: Default::default
///
#[unstable(feature = "never_type", issue = "35121")]
mod prim_never {}

#[doc(primitive = "char")]
/// A character type.
///
/// The `char` type represents a single character. More specifically, since
/// 'character' isn't a well-defined concept in Unicode, `char` is a '[Unicode
/// scalar value]', which is similar to, but not the same as, a '[Unicode code
/// point]'.
///
/// [Unicode scalar value]: https://www.unicode.org/glossary/#unicode_scalar_value
/// [Unicode code point]: https://www.unicode.org/glossary/#code_point
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
#[doc = concat!("[`String`]: ", include_str!("../primitive_docs/string_string.md"))]
///
/// As always, remember that a human intuition for 'character' might not map to
/// Unicode's definitions. For example, despite looking similar, the 'é'
/// character is one Unicode code point while 'é' is two Unicode code points:
///
/// ```
/// let mut chars = "é".chars();
/// // U+00e9: 'latin small letter e with acute'
/// assert_eq!(Some('\u{00e9}'), chars.next());
/// assert_eq!(None, chars.next());
///
/// let mut chars = "é".chars();
/// // U+0065: 'latin small letter e'
/// assert_eq!(Some('\u{0065}'), chars.next());
/// // U+0301: 'combining acute accent'
/// assert_eq!(Some('\u{0301}'), chars.next());
/// assert_eq!(None, chars.next());
/// ```
///
/// This means that the contents of the first string above _will_ fit into a
/// `char` while the contents of the second string _will not_. Trying to create
/// a `char` literal with the contents of the second string gives an error:
///
/// ```text
/// error: character literal may only contain one codepoint: 'é'
/// let c = 'é';
///         ^^^
/// ```
///
/// Another implication of the 4-byte fixed size of a `char` is that
/// per-`char` processing can end up using a lot more memory:
///
/// ```
/// let s = String::from("love: ❤️");
/// let v: Vec<char> = s.chars().collect();
///
/// assert_eq!(12, std::mem::size_of_val(&s[..]));
/// assert_eq!(32, std::mem::size_of_val(&v[..]));
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_char {}

#[doc(primitive = "unit")]
#[doc(alias = "(")]
#[doc(alias = ")")]
#[doc(alias = "()")]
//
/// The `()` type, also called "unit".
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
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_unit {}

#[doc(primitive = "pointer")]
#[doc(alias = "ptr")]
#[doc(alias = "*")]
#[doc(alias = "*const")]
#[doc(alias = "*mut")]
//
/// Raw, unsafe pointers, `*const T`, and `*mut T`.
///
/// *[See also the `std::ptr` module](ptr).*
///
/// Working with raw pointers in Rust is uncommon, typically limited to a few patterns.
/// Raw pointers can be unaligned or [`null`]. However, when a raw pointer is
/// dereferenced (using the `*` operator), it must be non-null and aligned.
///
/// Storing through a raw pointer using `*ptr = data` calls `drop` on the old value, so
/// [`write`] must be used if the type has drop glue and memory is not already
/// initialized - otherwise `drop` would be called on the uninitialized memory.
///
/// Use the [`null`] and [`null_mut`] functions to create null pointers, and the
/// [`is_null`] method of the `*const T` and `*mut T` types to check for null.
/// The `*const T` and `*mut T` types also define the [`offset`] method, for
/// pointer math.
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
/// The [`into_raw`] function consumes a box and returns
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
/// Note that here the call to [`drop`] is for clarity - it indicates
/// that we are done with the given value and it should be destroyed.
///
/// ## 3. Create it using `ptr::addr_of!`
///
/// Instead of coercing a reference to a raw pointer, you can use the macros
/// [`ptr::addr_of!`] (for `*const T`) and [`ptr::addr_of_mut!`] (for `*mut T`).
/// These macros allow you to create raw pointers to fields to which you cannot
/// create a reference (without causing undefined behaviour), such as an
/// unaligned field. This might be necessary if packed structs or uninitialized
/// memory is involved.
///
/// ```
/// #[derive(Debug, Default, Copy, Clone)]
/// #[repr(C, packed)]
/// struct S {
///     aligned: u8,
///     unaligned: u32,
/// }
/// let s = S::default();
/// let p = std::ptr::addr_of!(s.unaligned); // not allowed with coercion
/// ```
///
/// ## 4. Get it from C.
///
/// ```
/// # #![feature(rustc_private)]
/// extern crate libc;
///
/// use std::mem;
///
/// unsafe {
///     let my_num: *mut i32 = libc::malloc(mem::size_of::<i32>()) as *mut i32;
///     if my_num.is_null() {
///         panic!("failed to allocate memory");
///     }
///     libc::free(my_num as *mut libc::c_void);
/// }
/// ```
///
/// Usually you wouldn't literally use `malloc` and `free` from Rust,
/// but C APIs hand out a lot of pointers generally, so are a common source
/// of raw pointers in Rust.
///
/// [`null`]: ptr::null
/// [`null_mut`]: ptr::null_mut
/// [`is_null`]: pointer::is_null
/// [`offset`]: pointer::offset
#[doc = concat!("[`into_raw`]: ", include_str!("../primitive_docs/box_into_raw.md"))]
/// [`drop`]: mem::drop
/// [`write`]: ptr::write
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_pointer {}

#[doc(primitive = "array")]
#[doc(alias = "[]")]
#[doc(alias = "[T;N]")] // unfortunately, rustdoc doesn't have fuzzy search for aliases
#[doc(alias = "[T; N]")]
/// A fixed-size array, denoted `[T; N]`, for the element type, `T`, and the
/// non-negative compile-time constant size, `N`.
///
/// There are two syntactic forms for creating an array:
///
/// * A list with each element, i.e., `[x, y, z]`.
/// * A repeat expression `[x; N]`, which produces an array with `N` copies of `x`.
///   The type of `x` must be [`Copy`].
///
/// Note that `[expr; 0]` is allowed, and produces an empty array.
/// This will still evaluate `expr`, however, and immediately drop the resulting value, so
/// be mindful of side effects.
///
/// Arrays of *any* size implement the following traits if the element type allows it:
///
/// - [`Copy`]
/// - [`Clone`]
/// - [`Debug`]
/// - [`IntoIterator`] (implemented for `[T; N]`, `&[T; N]` and `&mut [T; N]`)
/// - [`PartialEq`], [`PartialOrd`], [`Eq`], [`Ord`]
/// - [`Hash`]
/// - [`AsRef`], [`AsMut`]
/// - [`Borrow`], [`BorrowMut`]
///
/// Arrays of sizes from 0 to 32 (inclusive) implement the [`Default`] trait
/// if the element type allows it. As a stopgap, trait implementations are
/// statically generated up to size 32.
///
/// Arrays coerce to [slices (`[T]`)][slice], so a slice method may be called on
/// an array. Indeed, this provides most of the API for working with arrays.
/// Slices have a dynamic size and do not coerce to arrays.
///
/// You can move elements out of an array with a [slice pattern]. If you want
/// one element, see [`mem::replace`].
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
/// for x in array {
///     print!("{} ", x);
/// }
/// ```
///
/// You can also iterate over reference to the array's elements:
///
/// ```
/// let array: [i32; 3] = [0; 3];
///
/// for x in &array { }
/// ```
///
/// You can use a [slice pattern] to move elements out of an array:
///
/// ```
/// fn move_away(_: String) { /* Do interesting things. */ }
///
/// let [john, roa] = ["John".to_string(), "Roa".to_string()];
/// move_away(john);
/// move_away(roa);
/// ```
///
/// # Editions
///
/// Prior to Rust 1.53, arrays did not implement [`IntoIterator`] by value, so the method call
/// `array.into_iter()` auto-referenced into a [slice iterator](slice::iter). Right now, the old
/// behavior is preserved in the 2015 and 2018 editions of Rust for compatibility, ignoring
/// [`IntoIterator`] by value. In the future, the behavior on the 2015 and 2018 edition
/// might be made consistent to the behavior of later editions.
///
/// ```rust,edition2018
/// // Rust 2015 and 2018:
///
/// # #![allow(array_into_iter)] // override our `deny(warnings)`
/// let array: [i32; 3] = [0; 3];
///
/// // This creates a slice iterator, producing references to each value.
/// for item in array.into_iter().enumerate() {
///     let (i, x): (usize, &i32) = item;
///     println!("array[{}] = {}", i, x);
/// }
///
/// // The `array_into_iter` lint suggests this change for future compatibility:
/// for item in array.iter().enumerate() {
///     let (i, x): (usize, &i32) = item;
///     println!("array[{}] = {}", i, x);
/// }
///
/// // You can explicitly iterate an array by value using
/// // `IntoIterator::into_iter` or `std::array::IntoIter::new`:
/// for item in IntoIterator::into_iter(array).enumerate() {
///     let (i, x): (usize, i32) = item;
///     println!("array[{}] = {}", i, x);
/// }
/// ```
///
/// Starting in the 2021 edition, `array.into_iter()` uses `IntoIterator` normally to iterate
/// by value, and `iter()` should be used to iterate by reference like previous editions.
///
/// ```rust,edition2021
/// // Rust 2021:
///
/// let array: [i32; 3] = [0; 3];
///
/// // This iterates by reference:
/// for item in array.iter().enumerate() {
///     let (i, x): (usize, &i32) = item;
///     println!("array[{}] = {}", i, x);
/// }
///
/// // This iterates by value:
/// for item in array.into_iter().enumerate() {
///     let (i, x): (usize, i32) = item;
///     println!("array[{}] = {}", i, x);
/// }
/// ```
///
/// Future language versions might start treating the `array.into_iter()`
/// syntax on editions 2015 and 2018 the same as on edition 2021. So code using
/// those older editions should still be written with this change in mind, to
/// prevent breakage in the future. The safest way to accomplish this is to
/// avoid the `into_iter` syntax on those editions. If an edition update is not
/// viable/desired, there are multiple alternatives:
/// * use `iter`, equivalent to the old behavior, creating references
/// * use [`IntoIterator::into_iter`], equivalent to the post-2021 behavior (Rust 1.53+)
/// * replace `for ... in array.into_iter() {` with `for ... in array {`,
///   equivalent to the post-2021 behavior (Rust 1.53+)
///
/// ```rust,edition2018
/// // Rust 2015 and 2018:
///
/// let array: [i32; 3] = [0; 3];
///
/// // This iterates by reference:
/// for item in array.iter() {
///     let x: &i32 = item;
///     println!("{}", x);
/// }
///
/// // This iterates by value:
/// for item in IntoIterator::into_iter(array) {
///     let x: i32 = item;
///     println!("{}", x);
/// }
///
/// // This iterates by value:
/// for item in array {
///     let x: i32 = item;
///     println!("{}", x);
/// }
///
/// // IntoIter can also start a chain.
/// // This iterates by value:
/// for item in IntoIterator::into_iter(array).enumerate() {
///     let (i, x): (usize, i32) = item;
///     println!("array[{}] = {}", i, x);
/// }
/// ```
///
/// [slice]: prim@slice
/// [`Debug`]: fmt::Debug
/// [`Hash`]: hash::Hash
/// [`Borrow`]: borrow::Borrow
/// [`BorrowMut`]: borrow::BorrowMut
/// [slice pattern]: ../reference/patterns.html#slice-patterns
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_array {}

#[doc(primitive = "slice")]
#[doc(alias = "[")]
#[doc(alias = "]")]
#[doc(alias = "[]")]
/// A dynamically-sized view into a contiguous sequence, `[T]`. Contiguous here
/// means that elements are laid out so that every element is the same
/// distance from its neighbors.
///
/// *[See also the `std::slice` module](crate::slice).*
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
/// let mut x = [1, 2, 3];
/// let x = &mut x[..]; // Take a full slice of `x`.
/// x[1] = 7;
/// assert_eq!(x, &[1, 7, 3]);
/// ```
///
/// As slices store the length of the sequence they refer to, they have twice
/// the size of pointers to [`Sized`](marker/trait.Sized.html) types.
/// Also see the reference on
/// [dynamically sized types](../reference/dynamically-sized-types.html).
///
/// ```
/// # use std::rc::Rc;
/// let pointer_size = std::mem::size_of::<&u8>();
/// assert_eq!(2 * pointer_size, std::mem::size_of::<&[u8]>());
/// assert_eq!(2 * pointer_size, std::mem::size_of::<*const [u8]>());
/// assert_eq!(2 * pointer_size, std::mem::size_of::<Box<[u8]>>());
/// assert_eq!(2 * pointer_size, std::mem::size_of::<Rc<[u8]>>());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_slice {}

#[doc(primitive = "str")]
//
/// String slices.
///
/// *[See also the `std::str` module](crate::str).*
///
/// The `str` type, also called a 'string slice', is the most primitive string
/// type. It is usually seen in its borrowed form, `&str`. It is also the type
/// of string literals, `&'static str`.
///
/// String slices are always valid UTF-8.
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
/// length. You can look at these with the [`as_ptr`] and [`len`] methods:
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
/// [`as_ptr`]: str::as_ptr
/// [`len`]: str::len
///
/// Note: This example shows the internals of `&str`. `unsafe` should not be
/// used to get a string slice under normal circumstances. Use `as_str`
/// instead.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_str {}

#[doc(primitive = "tuple")]
#[doc(alias = "(")]
#[doc(alias = ")")]
#[doc(alias = "()")]
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
/// ```
/// # let _:
/// (&'static str, i32, char)
/// # = ("hello", 5, 'c');
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
/// The sequential nature of the tuple applies to its implementations of various
/// traits. For example, in [`PartialOrd`] and [`Ord`], the elements are compared
/// sequentially until the first non-equal set is found.
///
/// For more about tuples, see [the book](../book/ch03-02-data-types.html#the-tuple-type).
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
/// [`Debug`]: fmt::Debug
/// [`Hash`]: hash::Hash
///
/// Due to a temporary restriction in Rust's type system, these traits are only
/// implemented on tuples of arity 12 or less. In the future, this may change.
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
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_tuple {}

#[doc(primitive = "f32")]
/// A 32-bit floating point type (specifically, the "binary32" type defined in IEEE 754-2008).
///
/// This type can represent a wide range of decimal numbers, like `3.5`, `27`,
/// `-113.75`, `0.0078125`, `34359738368`, `0`, `-1`. So unlike integer types
/// (such as `i32`), floating point types can represent non-integer numbers,
/// too.
///
/// However, being able to represent this wide range of numbers comes at the
/// cost of precision: floats can only represent some of the real numbers and
/// calculation with floats round to a nearby representable number. For example,
/// `5.0` and `1.0` can be exactly represented as `f32`, but `1.0 / 5.0` results
/// in `0.20000000298023223876953125` since `0.2` cannot be exactly represented
/// as `f32`. Note, however, that printing floats with `println` and friends will
/// often discard insignificant digits: `println!("{}", 1.0f32 / 5.0f32)` will
/// print `0.2`.
///
/// Additionally, `f32` can represent some special values:
///
/// - −0.0: IEEE 754 floating point numbers have a bit that indicates their sign, so −0.0 is a
///   possible value. For comparison −0.0 = +0.0, but floating point operations can carry
///   the sign bit through arithmetic operations. This means −0.0 × +0.0 produces −0.0 and
///   a negative number rounded to a value smaller than a float can represent also produces −0.0.
/// - [∞](#associatedconstant.INFINITY) and
///   [−∞](#associatedconstant.NEG_INFINITY): these result from calculations
///   like `1.0 / 0.0`.
/// - [NaN (not a number)](#associatedconstant.NAN): this value results from
///   calculations like `(-1.0).sqrt()`. NaN has some potentially unexpected
///   behavior: it is unequal to any float, including itself! It is also neither
///   smaller nor greater than any float, making it impossible to sort. Lastly,
///   it is considered infectious as almost all calculations where one of the
///   operands is NaN will also result in NaN.
///
/// For more information on floating point numbers, see [Wikipedia][wikipedia].
///
/// *[See also the `std::f32::consts` module](crate::f32::consts).*
///
/// [wikipedia]: https://en.wikipedia.org/wiki/Single-precision_floating-point_format
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_f32 {}

#[doc(primitive = "f64")]
/// A 64-bit floating point type (specifically, the "binary64" type defined in IEEE 754-2008).
///
/// This type is very similar to [`f32`], but has increased
/// precision by using twice as many bits. Please see [the documentation for
/// `f32`][`f32`] or [Wikipedia on double precision
/// values][wikipedia] for more information.
///
/// *[See also the `std::f64::consts` module](crate::f64::consts).*
///
/// [`f32`]: prim@f32
/// [wikipedia]: https://en.wikipedia.org/wiki/Double-precision_floating-point_format
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_f64 {}

#[doc(primitive = "i8")]
//
/// The 8-bit signed integer type.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_i8 {}

#[doc(primitive = "i16")]
//
/// The 16-bit signed integer type.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_i16 {}

#[doc(primitive = "i32")]
//
/// The 32-bit signed integer type.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_i32 {}

#[doc(primitive = "i64")]
//
/// The 64-bit signed integer type.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_i64 {}

#[doc(primitive = "i128")]
//
/// The 128-bit signed integer type.
#[stable(feature = "i128", since = "1.26.0")]
mod prim_i128 {}

#[doc(primitive = "u8")]
//
/// The 8-bit unsigned integer type.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_u8 {}

#[doc(primitive = "u16")]
//
/// The 16-bit unsigned integer type.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_u16 {}

#[doc(primitive = "u32")]
//
/// The 32-bit unsigned integer type.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_u32 {}

#[doc(primitive = "u64")]
//
/// The 64-bit unsigned integer type.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_u64 {}

#[doc(primitive = "u128")]
//
/// The 128-bit unsigned integer type.
#[stable(feature = "i128", since = "1.26.0")]
mod prim_u128 {}

#[doc(primitive = "isize")]
//
/// The pointer-sized signed integer type.
///
/// The size of this primitive is how many bytes it takes to reference any
/// location in memory. For example, on a 32 bit target, this is 4 bytes
/// and on a 64 bit target, this is 8 bytes.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_isize {}

#[doc(primitive = "usize")]
//
/// The pointer-sized unsigned integer type.
///
/// The size of this primitive is how many bytes it takes to reference any
/// location in memory. For example, on a 32 bit target, this is 4 bytes
/// and on a 64 bit target, this is 8 bytes.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_usize {}

#[doc(primitive = "reference")]
#[doc(alias = "&")]
#[doc(alias = "&mut")]
//
/// References, both shared and mutable.
///
/// A reference represents a borrow of some owned value. You can get one by using the `&` or `&mut`
/// operators on a value, or by using a [`ref`](../std/keyword.ref.html) or
/// <code>[ref](../std/keyword.ref.html) [mut](../std/keyword.mut.html)</code> pattern.
///
/// For those familiar with pointers, a reference is just a pointer that is assumed to be
/// aligned, not null, and pointing to memory containing a valid value of `T` - for example,
/// <code>&[bool]</code> can only point to an allocation containing the integer values `1`
/// ([`true`](../std/keyword.true.html)) or `0` ([`false`](../std/keyword.false.html)), but
/// creating a <code>&[bool]</code> that points to an allocation containing
/// the value `3` causes undefined behaviour.
/// In fact, <code>[Option]\<&T></code> has the same memory representation as a
/// nullable but aligned pointer, and can be passed across FFI boundaries as such.
///
/// In most cases, references can be used much like the original value. Field access, method
/// calling, and indexing work the same (save for mutability rules, of course). In addition, the
/// comparison operators transparently defer to the referent's implementation, allowing references
/// to be compared the same as owned values.
///
/// References have a lifetime attached to them, which represents the scope for which the borrow is
/// valid. A lifetime is said to "outlive" another one if its representative scope is as long or
/// longer than the other. The `'static` lifetime is the longest lifetime, which represents the
/// total life of the program. For example, string literals have a `'static` lifetime because the
/// text data is embedded into the binary of the program, rather than in an allocation that needs
/// to be dynamically managed.
///
/// `&mut T` references can be freely coerced into `&T` references with the same referent type, and
/// references with longer lifetimes can be freely coerced into references with shorter ones.
///
/// Reference equality by address, instead of comparing the values pointed to, is accomplished via
/// implicit reference-pointer coercion and raw pointer equality via [`ptr::eq`], while
/// [`PartialEq`] compares values.
///
/// ```
/// use std::ptr;
///
/// let five = 5;
/// let other_five = 5;
/// let five_ref = &five;
/// let same_five_ref = &five;
/// let other_five_ref = &other_five;
///
/// assert!(five_ref == same_five_ref);
/// assert!(five_ref == other_five_ref);
///
/// assert!(ptr::eq(five_ref, same_five_ref));
/// assert!(!ptr::eq(five_ref, other_five_ref));
/// ```
///
/// For more information on how to use references, see [the book's section on "References and
/// Borrowing"][book-refs].
///
/// [book-refs]: ../book/ch04-02-references-and-borrowing.html
///
/// # Trait implementations
///
/// The following traits are implemented for all `&T`, regardless of the type of its referent:
///
/// * [`Copy`]
/// * [`Clone`] \(Note that this will not defer to `T`'s `Clone` implementation if it exists!)
/// * [`Deref`]
/// * [`Borrow`]
/// * [`fmt::Pointer`]
///
/// [`Deref`]: ops::Deref
/// [`Borrow`]: borrow::Borrow
/// [`fmt::Pointer`]: fmt::Pointer
///
/// `&mut T` references get all of the above except `Copy` and `Clone` (to prevent creating
/// multiple simultaneous mutable borrows), plus the following, regardless of the type of its
/// referent:
///
/// * [`DerefMut`]
/// * [`BorrowMut`]
///
/// [`DerefMut`]: ops::DerefMut
/// [`BorrowMut`]: borrow::BorrowMut
/// [bool]: prim@bool
///
/// The following traits are implemented on `&T` references if the underlying `T` also implements
/// that trait:
///
/// * All the traits in [`std::fmt`] except [`fmt::Pointer`] (which is implemented regardless of the type of its referent) and [`fmt::Write`]
/// * [`PartialOrd`]
/// * [`Ord`]
/// * [`PartialEq`]
/// * [`Eq`]
/// * [`AsRef`]
/// * [`Fn`] \(in addition, `&T` references get [`FnMut`] and [`FnOnce`] if `T: Fn`)
/// * [`Hash`]
/// * [`ToSocketAddrs`]
/// * [`Send`] \(`&T` references also require <code>T: [Sync]</code>)
///
/// [`std::fmt`]: fmt
/// [`fmt::Pointer`]: fmt::Pointer
/// [`Hash`]: hash::Hash
#[doc = concat!("[`ToSocketAddrs`]: ", include_str!("../primitive_docs/net_tosocketaddrs.md"))]
///
/// `&mut T` references get all of the above except `ToSocketAddrs`, plus the following, if `T`
/// implements that trait:
///
/// * [`AsMut`]
/// * [`FnMut`] \(in addition, `&mut T` references get [`FnOnce`] if `T: FnMut`)
/// * [`fmt::Write`]
/// * [`Iterator`]
/// * [`DoubleEndedIterator`]
/// * [`ExactSizeIterator`]
/// * [`FusedIterator`]
/// * [`TrustedLen`]
/// * [`io::Write`]
/// * [`Read`]
/// * [`Seek`]
/// * [`BufRead`]
///
/// [`FusedIterator`]: iter::FusedIterator
/// [`TrustedLen`]: iter::TrustedLen
#[doc = concat!("[`Seek`]: ", include_str!("../primitive_docs/io_seek.md"))]
#[doc = concat!("[`BufRead`]: ", include_str!("../primitive_docs/io_bufread.md"))]
#[doc = concat!("[`Read`]: ", include_str!("../primitive_docs/io_read.md"))]
#[doc = concat!("[`io::Write`]: ", include_str!("../primitive_docs/io_write.md"))]
///
/// Note that due to method call deref coercion, simply calling a trait method will act like they
/// work on references as well as they do on owned values! The implementations described here are
/// meant for generic contexts, where the final type `T` is a type parameter or otherwise not
/// locally known.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_ref {}

#[doc(primitive = "fn")]
//
/// Function pointers, like `fn(usize) -> bool`.
///
/// *See also the traits [`Fn`], [`FnMut`], and [`FnOnce`].*
///
/// [`Fn`]: ops::Fn
/// [`FnMut`]: ops::FnMut
/// [`FnOnce`]: ops::FnOnce
///
/// Function pointers are pointers that point to *code*, not data. They can be called
/// just like functions. Like references, function pointers are, among other things, assumed to
/// not be null, so if you want to pass a function pointer over FFI and be able to accommodate null
/// pointers, make your type [`Option<fn()>`](core::option#options-and-pointers-nullable-pointers)
/// with your required signature.
///
/// ### Safety
///
/// Plain function pointers are obtained by casting either plain functions, or closures that don't
/// capture an environment:
///
/// ```
/// fn add_one(x: usize) -> usize {
///     x + 1
/// }
///
/// let ptr: fn(usize) -> usize = add_one;
/// assert_eq!(ptr(5), 6);
///
/// let clos: fn(usize) -> usize = |x| x + 5;
/// assert_eq!(clos(5), 10);
/// ```
///
/// In addition to varying based on their signature, function pointers come in two flavors: safe
/// and unsafe. Plain `fn()` function pointers can only point to safe functions,
/// while `unsafe fn()` function pointers can point to safe or unsafe functions.
///
/// ```
/// fn add_one(x: usize) -> usize {
///     x + 1
/// }
///
/// unsafe fn add_one_unsafely(x: usize) -> usize {
///     x + 1
/// }
///
/// let safe_ptr: fn(usize) -> usize = add_one;
///
/// //ERROR: mismatched types: expected normal fn, found unsafe fn
/// //let bad_ptr: fn(usize) -> usize = add_one_unsafely;
///
/// let unsafe_ptr: unsafe fn(usize) -> usize = add_one_unsafely;
/// let really_safe_ptr: unsafe fn(usize) -> usize = add_one;
/// ```
///
/// ### ABI
///
/// On top of that, function pointers can vary based on what ABI they use. This
/// is achieved by adding the `extern` keyword before the type, followed by the
/// ABI in question. The default ABI is "Rust", i.e., `fn()` is the exact same
/// type as `extern "Rust" fn()`. A pointer to a function with C ABI would have
/// type `extern "C" fn()`.
///
/// `extern "ABI" { ... }` blocks declare functions with ABI "ABI". The default
/// here is "C", i.e., functions declared in an `extern {...}` block have "C"
/// ABI.
///
/// For more information and a list of supported ABIs, see [the nomicon's
/// section on foreign calling conventions][nomicon-abi].
///
/// [nomicon-abi]: ../nomicon/ffi.html#foreign-calling-conventions
///
/// ### Variadic functions
///
/// Extern function declarations with the "C" or "cdecl" ABIs can also be *variadic*, allowing them
/// to be called with a variable number of arguments. Normal Rust functions, even those with an
/// `extern "ABI"`, cannot be variadic. For more information, see [the nomicon's section on
/// variadic functions][nomicon-variadic].
///
/// [nomicon-variadic]: ../nomicon/ffi.html#variadic-functions
///
/// ### Creating function pointers
///
/// When `bar` is the name of a function, then the expression `bar` is *not* a
/// function pointer. Rather, it denotes a value of an unnameable type that
/// uniquely identifies the function `bar`. The value is zero-sized because the
/// type already identifies the function. This has the advantage that "calling"
/// the value (it implements the `Fn*` traits) does not require dynamic
/// dispatch.
///
/// This zero-sized type *coerces* to a regular function pointer. For example:
///
/// ```rust
/// use std::mem;
///
/// fn bar(x: i32) {}
///
/// let not_bar_ptr = bar; // `not_bar_ptr` is zero-sized, uniquely identifying `bar`
/// assert_eq!(mem::size_of_val(&not_bar_ptr), 0);
///
/// let bar_ptr: fn(i32) = not_bar_ptr; // force coercion to function pointer
/// assert_eq!(mem::size_of_val(&bar_ptr), mem::size_of::<usize>());
///
/// let footgun = &bar; // this is a shared reference to the zero-sized type identifying `bar`
/// ```
///
/// The last line shows that `&bar` is not a function pointer either. Rather, it
/// is a reference to the function-specific ZST. `&bar` is basically never what you
/// want when `bar` is a function.
///
/// ### Traits
///
/// Function pointers implement the following traits:
///
/// * [`Clone`]
/// * [`PartialEq`]
/// * [`Eq`]
/// * [`PartialOrd`]
/// * [`Ord`]
/// * [`Hash`]
/// * [`Pointer`]
/// * [`Debug`]
///
/// [`Hash`]: hash::Hash
/// [`Pointer`]: fmt::Pointer
///
/// Due to a temporary restriction in Rust's type system, these traits are only implemented on
/// functions that take 12 arguments or less, with the `"Rust"` and `"C"` ABIs. In the future, this
/// may change.
///
/// In addition, function pointers of *any* signature, ABI, or safety are [`Copy`], and all *safe*
/// function pointers implement [`Fn`], [`FnMut`], and [`FnOnce`]. This works because these traits
/// are specially known to the compiler.
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_fn {}
