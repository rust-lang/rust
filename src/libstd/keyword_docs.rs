// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(keyword = "fn")]
//
/// The `fn` keyword.
///
/// The `fn` keyword is used to declare a function.
///
/// Example:
///
/// ```rust
/// fn some_function() {
///     // code goes in here
/// }
/// ```
///
/// For more information about functions, take a look at the [Rust Book][book].
///
/// [book]: https://doc.rust-lang.org/book/second-edition/ch03-03-how-functions-work.html
mod fn_keyword { }

#[doc(keyword = "let")]
//
/// The `let` keyword.
///
/// The `let` keyword is used to declare a variable.
///
/// Example:
///
/// ```rust
/// # #![allow(unused_assignments)]
/// let x = 3; // We create a variable named `x` with the value `3`.
/// ```
///
/// By default, all variables are **not** mutable. If you want a mutable variable,
/// you'll have to use the `mut` keyword.
///
/// Example:
///
/// ```rust
/// # #![allow(unused_assignments)]
/// let mut x = 3; // We create a mutable variable named `x` with the value `3`.
///
/// x += 4; // `x` is now equal to `7`.
/// ```
///
/// For more information about the `let` keyword, take a look at the [Rust Book][book].
///
/// [book]: https://doc.rust-lang.org/book/second-edition/ch03-01-variables-and-mutability.html
mod let_keyword { }

#[doc(keyword = "struct")]
//
/// The keyword used to define structs.
///
/// Structs in Rust come in three flavours: Regular structs, tuple structs,
/// and empty structs.
///
/// ```rust
/// struct Regular {
///     field1: f32,
///     field2: String,
///     pub field3: bool
/// }
///
/// struct Tuple(u32, String);
///
/// struct Empty;
/// ```
///
/// Regular structs are the most commonly used. Each field defined within them has a name and a
/// type, and once defined can be accessed using `example_struct.field` syntax. The fields of a
/// struct share its mutability, so `foo.bar = 2;` would only be valid if `foo` was mutable. Adding
/// `pub` to a field makes it visible to code in other modules, as well as allowing it to be
/// directly accessed and modified.
///
/// Tuple structs are similar to regular structs, but its fields have no names. They are used like
/// tuples, with deconstruction possible via `let TupleStruct(x, y) = foo;` syntax.  For accessing
/// individual variables, the same syntax is used as with regular tuples, namely `foo.0`, `foo.1`,
/// etc, starting at zero.
///
/// Empty structs, or unit-like structs, are most commonly used as markers, for example
/// [`PhantomData`]. Empty structs have a size of zero bytes, but unlike empty enums they can be
/// instantiated, making them similar to the unit type `()`. Unit-like structs are useful when you
/// need to implement a trait on something, but don't need to store any data inside it.
///
/// # Instantiation
///
/// Structs can be instantiated in a manner of different ways, each of which can be mixed and
/// matched as needed. The most common way to make a new struct is via a constructor method such as
/// `new()`, but when that isn't available (or you're writing the constructor itself), struct
/// literal syntax is used:
///
/// ```rust
/// # struct Foo { field1: f32, field2: String, etc: bool }
/// let example = Foo {
///     field1: 42.0,
///     field2: "blah".to_string(),
///     etc: true,
/// };
/// ```
///
/// It's only possible to directly instantiate a struct using struct literal syntax when all of its
/// fields are visible to you.
///
/// There are a handful of shortcuts provided to make writing constructors more convenient, most
/// common of which is the Field Init shorthand. When there is a variable and a field of the same
/// name, the assignment can be simplified from `field: field` into simply `field`. The following
/// example of a hypothetical constructor demonstrates this:
///
/// ```rust
/// struct User {
///     name: String,
///     admin: bool,
/// }
///
/// impl User {
///     pub fn new(name: String) -> Self {
///         Self {
///             name,
///             admin: false,
///         }
///     }
/// }
/// ```
///
/// Another shortcut for struct instantiation is available when you need to make a new struct that
/// shares most of a previous struct's values called struct update syntax:
///
/// ```rust
/// # struct Foo { field1: String, field2: () }
/// # let thing = Foo { field1: "".to_string(), field2: () };
/// let updated_thing = Foo {
///     field1: "a new value".to_string(),
///     ..thing
/// };
/// ```
///
/// Tuple structs are instantiated in the same way as tuples themselves, except with the struct's
/// name as a prefix: `Foo(123, false, 0.1)`.
///
/// Empty structs are instantiated with just their name and nothing else. `let thing =
/// EmptyStruct;`
///
///
/// # Style conventions
///
/// Structs are always written in CamelCase, with few exceptions. While the trailing comma on a
/// struct's list of fields can be omitted, it's usually kept for convenience in adding and
/// removing fields down the line.
///
/// For more information on structs, take a look at the [Rust Book][book] or the
/// [Reference][reference].
///
/// [`PhantomData`]: marker/struct.PhantomData.html
/// [book]: https://doc.rust-lang.org/book/second-edition/ch05-01-defining-structs.html
/// [reference]: https://doc.rust-lang.org/reference/items/structs.html

mod struct_keyword { }
