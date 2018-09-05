// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(keyword = "as")]
//
/// The type coercion keyword.
///
/// `as` is most commonly used to turn primitive types into other primitive types, but it has other
/// uses that include turning pointers into addresses, addresses into pointers, and pointers into
/// other pointers.
///
/// ```rust
/// let thing1: u8 = 89.0 as u8;
/// assert_eq!('B' as u32, 66);
/// assert_eq!(thing1 as char, 'Y');
/// let thing2: f32 = thing1 as f32 + 10.5;
/// assert_eq!(true as u8 + thing2 as u8, 100);
/// ```
///
/// In general, any coercion that can be performed via writing out type hints can also be done
/// using `as`, so instead of writing `let x: u32 = 123`, you can write `let x = 123 as u32` (Note:
/// `let x = 123u32` would be best in that situation). The same is not true in the other direction,
/// however, explicitly using `as` allows a few more coercions that aren't allowed implicitly, such
/// as changing the type of a raw pointer or turning closures into raw pointers.
///
/// For more information on what `as` is capable of, see the [Reference]
///
/// [Reference]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
mod as_keyword { }

#[doc(keyword = "const")]
//
/// The keyword for defining constants.
///
/// Sometimes a certain value is used many times throughout a program, and it can become
/// inconvenient to copy it over and over. What's more, it's not always possible or desirable to
/// make it a variable that gets carried around to each function that needs it. In these cases, the
/// `const` keyword provides a convenient alternative to code duplication.
///
/// ```rust
/// const THING: u32 = 0xABAD1DEA;
///
/// let foo = 123 + THING;
/// ```
///
/// Constants must be explicitly typed, unlike with `let` you can't ignore its type and let the
/// compiler figure it out. Any constant value can be defined in a const, which in practice happens
/// to be most things that would be reasonable to have a constant. For example, you can't have a
/// File as a `const`.
///
/// The only lifetime allowed in a constant is 'static, which is the lifetime that encompasses all
/// others in a Rust program. For example, if you wanted to define a constant string, it would look
/// like this:
///
/// ```rust
/// const WORDS: &'static str = "hello rust!";
/// ```
///
/// Thanks to static lifetime elision, you usually don't have to explicitly use 'static:
///
/// ```rust
/// const WORDS: &str = "hello convenience!";
/// ```
///
/// `const` items looks remarkably similar to [`static`] items, which introduces some confusion as
/// to which one should be used at which times. To put it simply, constants are inlined wherever
/// they're used, making using them identical to simply replacing the name of the const with its
/// value. Static variables on the other hand point to a single location in memory, which all
/// accesses share. This means that, unlike with constants, they can't have destructors, but it
/// also means that (via unsafe code) they can be mutable, which is useful for the rare situations
/// in which you can't avoid using global state.
///
/// Constants, as with statics, should always be in SCREAMING_SNAKE_CASE.
///
/// The `const` keyword is also used in raw pointers in combination with `mut`, as seen in `*const
/// T` and `*mut T`. More about that can be read at the [pointer] primitive part of the Rust docs.
///
/// For more detail on `const`, see the [Rust Book] or the [Reference]
///
/// [`static`]: keyword.static.html
/// [pointer]: primitive.pointer.html
/// [Rust Book]: https://doc.rust-lang.org/stable/book/2018-edition/ch03-01-variables-and-mutability.html#differences-between-variables-and-constants
/// [Reference]: https://doc.rust-lang.org/reference/items/constant-items.html
mod const_keyword { }

#[doc(keyword = "crate")]
//
/// The `crate` keyword.
///
/// The primary use of the `crate` keyword is as a part of `extern crate` declarations, which are
/// used to specify a dependency on a crate external to the one it's declared in. Crates are the
/// fundamental compilation unit of Rust code, and can be seen as libraries or projects. More can
/// be read about crates in the [Reference].
///
/// ```rust ignore
/// extern crate rand;
/// extern crate my_crate as thing;
/// extern crate std; // implicitly added to the root of every Rust project
/// ```
///
/// The `as` keyword can be used to change what the crate is referred to as in your project. If a
/// crate name includes a dash, it is implicitly imported with the dashes replaced by underscores.
///
/// `crate` is also used as in conjunction with [`pub`] to signify that the item it's attached to
/// is public only to other members of the same crate it's in.
///
/// ```rust
/// # #[allow(unused_imports)]
/// pub(crate) use std::io::Error as IoError;
/// pub(crate) enum CoolMarkerType { }
/// pub struct PublicThing {
///     pub(crate) semi_secret_thing: bool,
/// }
/// ```
///
/// [Reference]: https://doc.rust-lang.org/reference/items/extern-crates.html
/// [`pub`]: keyword.pub.html
mod crate_keyword { }

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
/// Another shortcut for struct instantiation is available, used when you need to make a new
/// struct that has the same values as most of a previous struct of the same type, called struct
/// update syntax:
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
/// Empty structs are instantiated with just their name, and don't need anything else. `let thing =
/// EmptyStruct;`
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
