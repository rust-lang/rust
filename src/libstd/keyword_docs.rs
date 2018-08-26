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
/// The `struct` keyword.
///
/// The `struct` keyword is used to define a struct type.
///
/// Example:
///
/// ```
/// struct Foo {
///     field1: u32,
///     field2: String,
/// }
/// ```
///
/// There are different kinds of structs. For more information, take a look at the
/// [Rust Book][book].
///
/// [book]: https://doc.rust-lang.org/book/second-edition/ch05-01-defining-structs.html
mod struct_keyword { }

#[doc(keyword = "enum")]
//
/// The `enum` keyword.
///
/// The `enum` keyword is used to define an enum type.
///
/// Example:
///
/// ```
/// enum Foo {
///     Value1,
///     Value2,
///     Value3(u32),
///     Value4 { x: u32, y: u64 },
/// }
/// ```
///
/// This is very convenient to handle different kind of data. To see which variant a value of an
/// enum type is of, you can use pattern matching on the value:
///
/// ```
/// enum Foo {
///     Value1,
///     Value2,
///     Value3(u32),
///     Value4 { x: u32, y: u64 },
/// }
///
/// let x = Foo::Value1;
///
/// match x {
///     Foo::Value1        => println!("This is Value1"),
///     Foo::Value2        => println!("This is Value2"),
///     Foo::Value3(_)     => println!("This is Value3"),
///     Foo::Value4 { .. } => println!("This is Value4"),
/// }
///
/// // Or:
///
/// if let Foo::Value1 = x {
///     println!("This is Value1");
/// } else {
///     println!("This not Value1");
/// }
/// ```
///
/// For more information, take a look at the [Rust Book][book].
///
/// [book]: https://doc.rust-lang.org/book/second-edition/ch06-01-defining-an-enum.html
mod enum_keyword { }
