#[doc(keyword = "as")]
//
/// Cast between types, or rename an import.
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
/// In general, any cast that can be performed via ascribing the type can also be done using `as`,
/// so instead of writing `let x: u32 = 123`, you can write `let x = 123 as u32` (Note: `let x: u32
/// = 123` would be best in that situation). The same is not true in the other direction, however,
/// explicitly using `as` allows a few more coercions that aren't allowed implicitly, such as
/// changing the type of a raw pointer or turning closures into raw pointers.
///
/// Other places `as` is used include as extra syntax for [`crate`] and `use`, to change the name
/// something is imported as.
///
/// For more information on what `as` is capable of, see the [Reference]
///
/// [Reference]: ../reference/expressions/operator-expr.html#type-cast-expressions
/// [`crate`]: keyword.crate.html
mod as_keyword { }

#[doc(keyword = "break")]
//
/// Exit early from a loop.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod break_keyword { }

#[doc(keyword = "const")]
//
/// Compile-time constants and deterministic functions.
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
/// to be most things that would be reasonable to have a constant (barring `const fn`s). For
/// example, you can't have a File as a `const`.
///
/// The only lifetime allowed in a constant is `'static`, which is the lifetime that encompasses
/// all others in a Rust program. For example, if you wanted to define a constant string, it would
/// look like this:
///
/// ```rust
/// const WORDS: &str = "hello rust!";
/// ```
///
/// Thanks to static lifetime elision, you usually don't have to explicitly use 'static:
///
/// ```rust
/// const WORDS: &str = "hello convenience!";
/// ```
///
/// `const` items looks remarkably similar to `static` items, which introduces some confusion as
/// to which one should be used at which times. To put it simply, constants are inlined wherever
/// they're used, making using them identical to simply replacing the name of the const with its
/// value. Static variables on the other hand point to a single location in memory, which all
/// accesses share. This means that, unlike with constants, they can't have destructors, and act as
/// a single value across the entire codebase.
///
/// Constants, as with statics, should always be in SCREAMING_SNAKE_CASE.
///
/// The `const` keyword is also used in raw pointers in combination with `mut`, as seen in `*const
/// T` and `*mut T`. More about that can be read at the [pointer] primitive part of the Rust docs.
///
/// For more detail on `const`, see the [Rust Book] or the [Reference]
///
/// [pointer]: primitive.pointer.html
/// [Rust Book]:
/// ../book/ch03-01-variables-and-mutability.html#differences-between-variables-and-constants
/// [Reference]: ../reference/items/constant-items.html
mod const_keyword { }

#[doc(keyword = "continue")]
//
/// Skip to the next iteration of a loop.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod continue_keyword { }

#[doc(keyword = "crate")]
//
/// A Rust binary or library.
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
/// `crate` is also used as in conjunction with `pub` to signify that the item it's attached to
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
/// [Reference]: ../reference/items/extern-crates.html
mod crate_keyword { }

#[doc(keyword = "else")]
//
/// What to do when an [`if`] condition does not hold.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [`if`]: keyword.if.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod else_keyword { }

#[doc(keyword = "enum")]
//
/// A type that can be any one of several variants.
///
/// Enums in Rust are similar to those of other compiled languages like C, but have important
/// differences that make them considerably more powerful. What Rust calls enums are more commonly
/// known as [Algebraic Data Types][ADT] if you're coming from a functional programming background.
/// The important detail is that each enum variant can have data to go along with it.
///
/// ```rust
/// # struct Coord;
/// enum SimpleEnum {
///     FirstVariant,
///     SecondVariant,
///     ThirdVariant,
/// }
///
/// enum Location {
///     Unknown,
///     Anonymous,
///     Known(Coord),
/// }
///
/// enum ComplexEnum {
///     Nothing,
///     Something(u32),
///     LotsOfThings {
///         usual_struct_stuff: bool,
///         blah: String,
///     }
/// }
///
/// enum EmptyEnum { }
/// ```
///
/// The first enum shown is the usual kind of enum you'd find in a C-style language. The second
/// shows off a hypothetical example of something storing location data, with `Coord` being any
/// other type that's needed, for example a struct. The third example demonstrates the kind of
/// data a variant can store, ranging from nothing, to a tuple, to an anonymous struct.
///
/// Instantiating enum variants involves explicitly using the enum's name as its namespace,
/// followed by one of its variants. `SimpleEnum::SecondVariant` would be an example from above.
/// When data follows along with a variant, such as with rust's built-in [`Option`] type, the data
/// is added as the type describes, for example `Option::Some(123)`. The same follows with
/// struct-like variants, with things looking like `ComplexEnum::LotsOfThings { usual_struct_stuff:
/// true, blah: "hello!".to_string(), }`. Empty Enums are similar to () in that they cannot be
/// instantiated at all, and are used mainly to mess with the type system in interesting ways.
///
/// For more information, take a look at the [Rust Book] or the [Reference]
///
/// [ADT]: https://en.wikipedia.org/wiki/Algebraic_data_type
/// [`Option`]: option/enum.Option.html
/// [Rust Book]: ../book/ch06-01-defining-an-enum.html
/// [Reference]: ../reference/items/enumerations.html
mod enum_keyword { }

#[doc(keyword = "extern")]
//
/// Link to or import external code.
///
/// The `extern` keyword is used in two places in Rust. One is in conjunction with the [`crate`]
/// keyword to make your Rust code aware of other Rust crates in your project, i.e., `extern crate
/// lazy_static;`. The other use is in foreign function interfaces (FFI).
///
/// `extern` is used in two different contexts within FFI. The first is in the form of external
/// blocks, for declaring function interfaces that Rust code can call foreign code by.
///
/// ```rust ignore
/// #[link(name = "my_c_library")]
/// extern "C" {
///     fn my_c_function(x: i32) -> bool;
/// }
/// ```
///
/// This code would attempt to link with `libmy_c_library.so` on unix-like systems and
/// `my_c_library.dll` on Windows at runtime, and panic if it can't find something to link to. Rust
/// code could then use `my_c_function` as if it were any other unsafe Rust function. Working with
/// non-Rust languages and FFI is inherently unsafe, so wrappers are usually built around C APIs.
///
/// The mirror use case of FFI is also done via the `extern` keyword:
///
/// ```rust
/// #[no_mangle]
/// pub extern fn callable_from_c(x: i32) -> bool {
///     x % 3 == 0
/// }
/// ```
///
/// If compiled as a dylib, the resulting .so could then be linked to from a C library, and the
/// function could be used as if it was from any other library.
///
/// For more information on FFI, check the [Rust book] or the [Reference].
///
/// [Rust book]:
/// ../book/ch19-01-unsafe-rust.html#using-extern-functions-to-call-external-code
/// [Reference]: ../reference/items/external-blocks.html
mod extern_keyword { }

#[doc(keyword = "false")]
//
/// A value of type [`bool`] representing logical **false**.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [`bool`]: primitive.bool.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod false_keyword { }

#[doc(keyword = "fn")]
//
/// A function or function pointer.
///
/// Functions are the primary way code is executed within Rust. Function blocks, usually just
/// called functions, can be defined in a variety of different places and be assigned many
/// different attributes and modifiers.
///
/// Standalone functions that just sit within a module not attached to anything else are common,
/// but most functions will end up being inside [`impl`] blocks, either on another type itself, or
/// as a trait impl for that type.
///
/// ```rust
/// fn standalone_function() {
///     // code
/// }
///
/// pub fn public_thing(argument: bool) -> String {
///     // code
///     # "".to_string()
/// }
///
/// struct Thing {
///     foo: i32,
/// }
///
/// impl Thing {
///     pub fn new() -> Self {
///         Self {
///             foo: 42,
///         }
///     }
/// }
/// ```
///
/// In addition to presenting fixed types in the form of `fn name(arg: type, ..) -> return_type`,
/// functions can also declare a list of type parameters along with trait bounds that they fall
/// into.
///
/// ```rust
/// fn generic_function<T: Clone>(x: T) -> (T, T, T) {
///     (x.clone(), x.clone(), x.clone())
/// }
///
/// fn generic_where<T>(x: T) -> T
///     where T: std::ops::Add<Output = T> + Copy
/// {
///     x + x + x
/// }
/// ```
///
/// Declaring trait bounds in the angle brackets is functionally identical to using a `where`
/// clause. It's up to the programmer to decide which works better in each situation, but `where`
/// tends to be better when things get longer than one line.
///
/// Along with being made public via `pub`, `fn` can also have an [`extern`] added for use in
/// FFI.
///
/// For more information on the various types of functions and how they're used, consult the [Rust
/// book] or the [Reference].
///
/// [`impl`]: keyword.impl.html
/// [`extern`]: keyword.extern.html
/// [Rust book]: ../book/ch03-03-how-functions-work.html
/// [Reference]: ../reference/items/functions.html
mod fn_keyword { }

#[doc(keyword = "for")]
//
/// Iteration with [`in`], trait implementation with [`impl`], or [higher-ranked trait bounds]
/// (`for<'a>`).
///
/// The `for` keyword is used in many syntactic locations:
///
/// * `for` is used in for-in-loops (see below).
/// * `for` is used when implementing traits as in `impl Trait for Type` (see [`impl`] for more info
///   on that).
/// * `for` is also used for [higher-ranked trait bounds] as in `for<'a> &'a T: PartialEq<i32>`.
///
/// for-in-loops, or to be more precise, iterator loops, are a simple syntactic sugar over a common
/// practice within Rust, which is to loop over an iterator until that iterator returns `None` (or
/// `break` is called).
///
/// ```rust
/// for i in 0..5 {
///     println!("{}", i * 2);
/// }
///
/// for i in std::iter::repeat(5) {
///     println!("turns out {} never stops being 5", i);
///     break; // would loop forever otherwise
/// }
///
/// 'outer: for x in 5..50 {
///     for y in 0..10 {
///         if x == y {
///             break 'outer;
///         }
///     }
/// }
/// ```
///
/// As shown in the example above, `for` loops (along with all other loops) can be tagged, using
/// similar syntax to lifetimes (only visually similar, entirely distinct in practice). Giving the
/// same tag to `break` breaks the tagged loop, which is useful for inner loops. It is definitely
/// not a goto.
///
/// A `for` loop expands as shown:
///
/// ```rust
/// # fn code() { }
/// # let iterator = 0..2;
/// for loop_variable in iterator {
///     code()
/// }
/// ```
///
/// ```rust
/// # fn code() { }
/// # let iterator = 0..2;
/// {
///     let mut _iter = std::iter::IntoIterator::into_iter(iterator);
///     loop {
///         match _iter.next() {
///             Some(loop_variable) => {
///                 code()
///             },
///             None => break,
///         }
///     }
/// }
/// ```
///
/// More details on the functionality shown can be seen at the [`IntoIterator`] docs.
///
/// For more information on for-loops, see the [Rust book] or the [Reference].
///
/// [`in`]: keyword.in.html
/// [`impl`]: keyword.impl.html
/// [higher-ranked trait bounds]: ../reference/trait-bounds.html#higher-ranked-trait-bounds
/// [`IntoIterator`]: iter/trait.IntoIterator.html
/// [Rust book]:
/// ../book/ch03-05-control-flow.html#looping-through-a-collection-with-for
/// [Reference]: ../reference/expressions/loop-expr.html#iterator-loops
mod for_keyword { }

#[doc(keyword = "if")]
//
/// Evaluate a block if a condition holds.
///
/// `if` is a familiar construct to most programmers, and is the main way you'll often do logic in
/// your code. However, unlike in most languages, `if` blocks can also act as expressions.
///
/// ```rust
/// # let rude = true;
/// if 1 == 2 {
///     println!("whoops, mathematics broke");
/// } else {
///     println!("everything's fine!");
/// }
///
/// let greeting = if rude {
///     "sup nerd."
/// } else {
///     "hello, friend!"
/// };
///
/// if let Ok(x) = "123".parse::<i32>() {
///     println!("{} double that and you get {}!", greeting, x * 2);
/// }
/// ```
///
/// Shown above are the three typical forms an `if` block comes in. First is the usual kind of
/// thing you'd see in many languages, with an optional `else` block. Second uses `if` as an
/// expression, which is only possible if all branches return the same type. An `if` expression can
/// be used everywhere you'd expect. The third kind of `if` block is an `if let` block, which
/// behaves similarly to using a `match` expression:
///
/// ```rust
/// if let Some(x) = Some(123) {
///     // code
///     # let _ = x;
/// } else {
///     // something else
/// }
///
/// match Some(123) {
///     Some(x) => {
///         // code
///         # let _ = x;
///     },
///     _ => {
///         // something else
///     },
/// }
/// ```
///
/// Each kind of `if` expression can be mixed and matched as needed.
///
/// ```rust
/// if true == false {
///     println!("oh no");
/// } else if "something" == "other thing" {
///     println!("oh dear");
/// } else if let Some(200) = "blarg".parse::<i32>().ok() {
///     println!("uh oh");
/// } else {
///     println!("phew, nothing's broken");
/// }
/// ```
///
/// The `if` keyword is used in one other place in Rust, namely as a part of pattern matching
/// itself, allowing patterns such as `Some(x) if x > 200` to be used.
///
/// For more information on `if` expressions, see the [Rust book] or the [Reference].
///
/// [Rust book]: ../book/ch03-05-control-flow.html#if-expressions
/// [Reference]: ../reference/expressions/if-expr.html
mod if_keyword { }

#[doc(keyword = "impl")]
//
/// Implement some functionality for a type.
///
/// The `impl` keyword is primarily used to define implementations on types. Inherent
/// implementations are standalone, while trait implementations are used to implement traits for
/// types, or other traits.
///
/// Functions and consts can both be defined in an implementation. A function defined in an
/// `impl` block can be standalone, meaning it would be called like `Foo::bar()`. If the function
/// takes `self`, `&self`, or `&mut self` as its first argument, it can also be called using
/// method-call syntax, a familiar feature to any object oriented programmer, like `foo.bar()`.
///
/// ```rust
/// struct Example {
///     number: i32,
/// }
///
/// impl Example {
///     fn boo() {
///         println!("boo! Example::boo() was called!");
///     }
///
///     fn answer(&mut self) {
///         self.number += 42;
///     }
///
///     fn get_number(&self) -> i32 {
///         self.number
///     }
/// }
///
/// trait Thingy {
///     fn do_thingy(&self);
/// }
///
/// impl Thingy for Example {
///     fn do_thingy(&self) {
///         println!("doing a thing! also, number is {}!", self.number);
///     }
/// }
/// ```
///
/// For more information on implementations, see the [Rust book][book1] or the [Reference].
///
/// The other use of the `impl` keyword is in `impl Trait` syntax, which can be seen as a shorthand
/// for "a concrete type that implements this trait". Its primary use is working with closures,
/// which have type definitions generated at compile time that can't be simply typed out.
///
/// ```rust
/// fn thing_returning_closure() -> impl Fn(i32) -> bool {
///     println!("here's a closure for you!");
///     |x: i32| x % 3 == 0
/// }
/// ```
///
/// For more information on `impl Trait` syntax, see the [Rust book][book2].
///
/// [book1]: ../book/ch05-03-method-syntax.html
/// [Reference]: ../reference/items/implementations.html
/// [book2]: ../book/ch10-02-traits.html#returning-types-that-implement-traits
mod impl_keyword { }

#[doc(keyword = "in")]
//
/// Iterate over a series of values with [`for`].
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [`for`]: keyword.for.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod in_keyword { }

#[doc(keyword = "let")]
//
/// Bind a value to a variable.
///
/// The primary use for the `let` keyword is in `let` statements, which are used to introduce a new
/// set of variables into the current scope, as given by a pattern.
///
/// ```rust
/// # #![allow(unused_assignments)]
/// let thing1: i32 = 100;
/// let thing2 = 200 + thing1;
///
/// let mut changing_thing = true;
/// changing_thing = false;
///
/// let (part1, part2) = ("first", "second");
///
/// struct Example {
///     a: bool,
///     b: u64,
/// }
///
/// let Example { a, b: _ } = Example {
///     a: true,
///     b: 10004,
/// };
/// assert!(a);
/// ```
///
/// The pattern is most commonly a single variable, which means no pattern matching is done and
/// the expression given is bound to the variable. Apart from that, patterns used in `let` bindings
/// can be as complicated as needed, given that the pattern is exhaustive. See the [Rust
/// book][book1] for more information on pattern matching. The type of the pattern is optionally
/// given afterwards, but if left blank is automatically inferred by the compiler if possible.
///
/// Variables in Rust are immutable by default, and require the `mut` keyword to be made mutable.
///
/// Multiple variables can be defined with the same name, known as shadowing. This doesn't affect
/// the original variable in any way beyond being unable to directly access it beyond the point of
/// shadowing. It continues to remain in scope, getting dropped only when it falls out of scope.
/// Shadowed variables don't need to have the same type as the variables shadowing them.
///
/// ```rust
/// let shadowing_example = true;
/// let shadowing_example = 123.4;
/// let shadowing_example = shadowing_example as u32;
/// let mut shadowing_example = format!("cool! {}", shadowing_example);
/// shadowing_example += " something else!"; // not shadowing
/// ```
///
/// Other places the `let` keyword is used include along with [`if`], in the form of `if let`
/// expressions. They're useful if the pattern being matched isn't exhaustive, such as with
/// enumerations. `while let` also exists, which runs a loop with a pattern matched value until
/// that pattern can't be matched.
///
/// For more information on the `let` keyword, see the [Rust book][book2] or the [Reference]
///
/// [book1]: ../book/ch06-02-match.html
/// [`if`]: keyword.if.html
/// [book2]: ../book/ch18-01-all-the-places-for-patterns.html#let-statements
/// [Reference]: ../reference/statements.html#let-statements
mod let_keyword { }

#[doc(keyword = "loop")]
//
/// Loop indefinitely.
///
/// `loop` is used to define the simplest kind of loop supported in Rust. It runs the code inside
/// it until the code uses `break` or the program exits.
///
/// ```rust
/// loop {
///     println!("hello world forever!");
///     # break;
/// }
///
/// let mut i = 0;
/// loop {
///     println!("i is {}", i);
///     if i > 10 {
///         break;
///     }
///     i += 1;
/// }
/// ```
///
/// Unlike the other kinds of loops in Rust (`while`, `while let`, and `for`), loops can be used as
/// expressions that return values via `break`.
///
/// ```rust
/// let mut i = 1;
/// let something = loop {
///     i *= 2;
///     if i > 100 {
///         break i;
///     }
/// };
/// assert_eq!(something, 128);
/// ```
///
/// Every `break` in a loop has to have the same type. When it's not explicitly giving something,
/// `break;` returns `()`.
///
/// For more information on `loop` and loops in general, see the [Reference].
///
/// [Reference]: ../reference/expressions/loop-expr.html
mod loop_keyword { }

#[doc(keyword = "match")]
//
/// Control flow based on pattern matching.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod match_keyword { }

#[doc(keyword = "mod")]
//
/// Organize code into [modules].
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [modules]: ../reference/items/modules.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod mod_keyword { }

#[doc(keyword = "move")]
//
/// Capture a [closure]'s environment by value.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [closure]: ../book/second-edition/ch13-01-closures.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod move_keyword { }

#[doc(keyword = "mut")]
//
/// A mutable binding, reference, or pointer.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod mut_keyword { }

#[doc(keyword = "pub")]
//
/// Make an item visible to others.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod pub_keyword { }

#[doc(keyword = "ref")]
//
/// Bind by reference during pattern matching.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod ref_keyword { }

#[doc(keyword = "return")]
//
/// Return a value from a function.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod return_keyword { }

#[doc(keyword = "self")]
//
/// The receiver of a method, or the current module.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod self_keyword { }

#[doc(keyword = "Self")]
//
/// The implementing type within a [`trait`] or [`impl`] block, or the current type within a type
/// definition.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [`impl`]: keyword.impl.html
/// [`trait`]: keyword.trait.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod self_upper_keyword { }

#[doc(keyword = "static")]
//
/// A place that is valid for the duration of a program.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod static_keyword { }

#[doc(keyword = "struct")]
//
/// A type that is composed of other types.
///
/// Structs in Rust come in three flavors: Structs with named fields, tuple structs, and unit
/// structs.
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
/// struct Unit;
/// ```
///
/// Regular structs are the most commonly used. Each field defined within them has a name and a
/// type, and once defined can be accessed using `example_struct.field` syntax. The fields of a
/// struct share its mutability, so `foo.bar = 2;` would only be valid if `foo` was mutable. Adding
/// `pub` to a field makes it visible to code in other modules, as well as allowing it to be
/// directly accessed and modified.
///
/// Tuple structs are similar to regular structs, but its fields have no names. They are used like
/// tuples, with deconstruction possible via `let TupleStruct(x, y) = foo;` syntax. For accessing
/// individual variables, the same syntax is used as with regular tuples, namely `foo.0`, `foo.1`,
/// etc, starting at zero.
///
/// Unit structs are most commonly used as marker. They have a size of zero bytes, but unlike empty
/// enums they can be instantiated, making them isomorphic to the unit type `()`. Unit structs are
/// useful when you need to implement a trait on something, but don't need to store any data inside
/// it.
///
/// # Instantiation
///
/// Structs can be instantiated in different ways, all of which can be mixed and
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
/// [book]: ../book/ch05-01-defining-structs.html
/// [reference]: ../reference/items/structs.html
mod struct_keyword { }

#[doc(keyword = "super")]
//
/// The parent of the current [module].
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [module]: ../reference/items/modules.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod super_keyword { }

#[doc(keyword = "trait")]
//
/// A common interface for a class of types.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod trait_keyword { }

#[doc(keyword = "true")]
//
/// A value of type [`bool`] representing logical **true**.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [`bool`]: primitive.bool.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod true_keyword { }

#[doc(keyword = "type")]
//
/// Define an alias for an existing type.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod type_keyword { }

#[doc(keyword = "unsafe")]
//
/// Code or interfaces whose [memory safety] cannot be verified by the type system.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [memory safety]: ../book/ch19-01-unsafe-rust.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod unsafe_keyword { }

#[doc(keyword = "use")]
//
/// Import or rename items from other crates or modules.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod use_keyword { }

#[doc(keyword = "where")]
//
/// Add constraints that must be upheld to use an item.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod where_keyword { }

#[doc(keyword = "while")]
//
/// Loop while a condition is upheld.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod while_keyword { }

// 2018 Edition keywords

#[unstable(feature = "async_await", issue = "50547")]
#[doc(keyword = "async")]
//
/// Return a [`Future`] instead of blocking the current thread.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [`Future`]: ./future/trait.Future.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod async_keyword { }

#[unstable(feature = "async_await", issue = "50547")]
#[doc(keyword = "await")]
//
/// Suspend execution until the result of a [`Future`] is ready.
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [`Future`]: ./future/trait.Future.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod await_keyword { }

#[doc(keyword = "dyn")]
//
/// Name the type of a [trait object].
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [trait object]: ../book/ch17-02-trait-objects.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod dyn_keyword { }

#[doc(keyword = "union")]
//
/// The [Rust equivalent of a C-style union][union].
///
/// The documentation for this keyword is [not yet complete]. Pull requests welcome!
///
/// [union]: ../reference/items/unions.html
/// [not yet complete]: https://github.com/rust-lang/rust/issues/34601
mod union_keyword { }
