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
/// so instead of writing `let x: u32 = 123`, you can write `let x = 123 as u32` (note: `let x: u32
/// = 123` would be best in that situation). The same is not true in the other direction, however;
/// explicitly using `as` allows a few more coercions that aren't allowed implicitly, such as
/// changing the type of a raw pointer or turning closures into raw pointers.
///
/// `as` can be seen as the primitive for `From` and `Into`: `as` only works  with primitives
/// (`u8`, `bool`, `str`, pointers, ...) whereas `From` and `Into`  also works with types like
/// `String` or `Vec`.
///
/// `as` can also be used with the `_` placeholder when the destination type can be inferred. Note
/// that this can cause inference breakage and usually such code should use an explicit type for
/// both clarity and stability. This is most useful when converting pointers using `as *const _` or
/// `as *mut _` though the [`cast`][const-cast] method is recommended over `as *const _` and it is
/// [the same][mut-cast] for `as *mut _`: those methods make the intent clearer.
///
/// `as` is also used to rename imports in [`use`] and [`extern crate`][`crate`] statements:
///
/// ```
/// # #[allow(unused_imports)]
/// use std::{mem as memory, net as network};
/// // Now you can use the names `memory` and `network` to refer to `std::mem` and `std::net`.
/// ```
/// For more information on what `as` is capable of, see the [Reference].
///
/// [Reference]: ../reference/expressions/operator-expr.html#type-cast-expressions
/// [`crate`]: keyword.crate.html
/// [`use`]: keyword.use.html
/// [const-cast]: pointer::cast
/// [mut-cast]: primitive.pointer.html#method.cast-1
mod as_keyword {}

#[doc(keyword = "break")]
//
/// Exit early from a loop.
///
/// When `break` is encountered, execution of the associated loop body is
/// immediately terminated.
///
/// ```rust
/// let mut last = 0;
///
/// for x in 1..100 {
///     if x > 12 {
///         break;
///     }
///     last = x;
/// }
///
/// assert_eq!(last, 12);
/// println!("{}", last);
/// ```
///
/// A break expression is normally associated with the innermost loop enclosing the
/// `break` but a label can be used to specify which enclosing loop is affected.
///
///```rust
/// 'outer: for i in 1..=5 {
///     println!("outer iteration (i): {}", i);
///
///     '_inner: for j in 1..=200 {
///         println!("    inner iteration (j): {}", j);
///         if j >= 3 {
///             // breaks from inner loop, let's outer loop continue.
///             break;
///         }
///         if i >= 2 {
///             // breaks from outer loop, and directly to "Bye".
///             break 'outer;
///         }
///     }
/// }
/// println!("Bye.");
///```
///
/// When associated with `loop`, a break expression may be used to return a value from that loop.
/// This is only valid with `loop` and not with any other type of loop.
/// If no value is specified, `break;` returns `()`.
/// Every `break` within a loop must return the same type.
///
/// ```rust
/// let (mut a, mut b) = (1, 1);
/// let result = loop {
///     if b > 10 {
///         break b;
///     }
///     let c = a + b;
///     a = b;
///     b = c;
/// };
/// // first number in Fibonacci sequence over 10:
/// assert_eq!(result, 13);
/// println!("{}", result);
/// ```
///
/// For more details consult the [Reference on "break expression"] and the [Reference on "break and
/// loop values"].
///
/// [Reference on "break expression"]: ../reference/expressions/loop-expr.html#break-expressions
/// [Reference on "break and loop values"]:
/// ../reference/expressions/loop-expr.html#break-and-loop-values
mod break_keyword {}

#[doc(keyword = "const")]
//
/// Compile-time constants and compile-time evaluable functions.
///
/// ## Compile-time constants
///
/// Sometimes a certain value is used many times throughout a program, and it can become
/// inconvenient to copy it over and over. What's more, it's not always possible or desirable to
/// make it a variable that gets carried around to each function that needs it. In these cases, the
/// `const` keyword provides a convenient alternative to code duplication:
///
/// ```rust
/// const THING: u32 = 0xABAD1DEA;
///
/// let foo = 123 + THING;
/// ```
///
/// Constants must be explicitly typed; unlike with `let`, you can't ignore their type and let the
/// compiler figure it out. Any constant value can be defined in a `const`, which in practice happens
/// to be most things that would be reasonable to have in a constant (barring `const fn`s). For
/// example, you can't have a [`File`] as a `const`.
///
/// [`File`]: crate::fs::File
///
/// The only lifetime allowed in a constant is `'static`, which is the lifetime that encompasses
/// all others in a Rust program. For example, if you wanted to define a constant string, it would
/// look like this:
///
/// ```rust
/// const WORDS: &'static str = "hello rust!";
/// ```
///
/// Thanks to static lifetime elision, you usually don't have to explicitly use `'static`:
///
/// ```rust
/// const WORDS: &str = "hello convenience!";
/// ```
///
/// `const` items looks remarkably similar to `static` items, which introduces some confusion as
/// to which one should be used at which times. To put it simply, constants are inlined wherever
/// they're used, making using them identical to simply replacing the name of the `const` with its
/// value. Static variables, on the other hand, point to a single location in memory, which all
/// accesses share. This means that, unlike with constants, they can't have destructors, and act as
/// a single value across the entire codebase.
///
/// Constants, like statics, should always be in `SCREAMING_SNAKE_CASE`.
///
/// For more detail on `const`, see the [Rust Book] or the [Reference].
///
/// ## Compile-time evaluable functions
///
/// The other main use of the `const` keyword is in `const fn`. This marks a function as being
/// callable in the body of a `const` or `static` item and in array initializers (commonly called
/// "const contexts"). `const fn` are restricted in the set of operations they can perform, to
/// ensure that they can be evaluated at compile-time. See the [Reference][const-eval] for more
/// detail.
///
/// Turning a `fn` into a `const fn` has no effect on run-time uses of that function.
///
/// ## Other uses of `const`
///
/// The `const` keyword is also used in raw pointers in combination with `mut`, as seen in `*const
/// T` and `*mut T`. More about `const` as used in raw pointers can be read at the Rust docs for the [pointer primitive].
///
/// [pointer primitive]: pointer
/// [Rust Book]: ../book/ch03-01-variables-and-mutability.html#differences-between-variables-and-constants
/// [Reference]: ../reference/items/constant-items.html
/// [const-eval]: ../reference/const_eval.html
mod const_keyword {}

#[doc(keyword = "continue")]
//
/// Skip to the next iteration of a loop.
///
/// When `continue` is encountered, the current iteration is terminated, returning control to the
/// loop head, typically continuing with the next iteration.
///
///```rust
/// // Printing odd numbers by skipping even ones
/// for number in 1..=10 {
///     if number % 2 == 0 {
///         continue;
///     }
///     println!("{}", number);
/// }
///```
///
/// Like `break`, `continue` is normally associated with the innermost enclosing loop, but labels
/// may be used to specify the affected loop.
///
///```rust
/// // Print Odd numbers under 30 with unit <= 5
/// 'tens: for ten in 0..3 {
///     '_units: for unit in 0..=9 {
///         if unit % 2 == 0 {
///             continue;
///         }
///         if unit > 5 {
///             continue 'tens;
///         }
///         println!("{}", ten * 10 + unit);
///     }
/// }
///```
///
/// See [continue expressions] from the reference for more details.
///
/// [continue expressions]: ../reference/expressions/loop-expr.html#continue-expressions
mod continue_keyword {}

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
/// `crate` can also be used as in conjunction with `pub` to signify that the item it's attached to
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
/// `crate` is also used to represent the absolute path of a module, where `crate` refers to the
/// root of the current crate. For instance, `crate::foo::bar` refers to the name `bar` inside the
/// module `foo`, from anywhere else in the same crate.
///
/// [Reference]: ../reference/items/extern-crates.html
mod crate_keyword {}

#[doc(keyword = "else")]
//
/// What expression to evaluate when an [`if`] condition evaluates to [`false`].
///
/// `else` expressions are optional. When no else expressions are supplied it is assumed to evaluate
/// to the unit type `()`.
///
/// The type that the `else` blocks evaluate to must be compatible with the type that the `if` block
/// evaluates to.
///
/// As can be seen below, `else` must be followed by either: `if`, `if let`, or a block `{}` and it
/// will return the value of that expression.
///
/// ```rust
/// let result = if true == false {
///     "oh no"
/// } else if "something" == "other thing" {
///     "oh dear"
/// } else if let Some(200) = "blarg".parse::<i32>().ok() {
///     "uh oh"
/// } else {
///     println!("Sneaky side effect.");
///     "phew, nothing's broken"
/// };
/// ```
///
/// Here's another example but here we do not try and return an expression:
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
/// The above is _still_ an expression but it will always evaluate to `()`.
///
/// There is possibly no limit to the number of `else` blocks that could follow an `if` expression
/// however if you have several then a [`match`] expression might be preferable.
///
/// Read more about control flow in the [Rust Book].
///
/// [Rust Book]: ../book/ch03-05-control-flow.html#handling-multiple-conditions-with-else-if
/// [`match`]: keyword.match.html
/// [`false`]: keyword.false.html
/// [`if`]: keyword.if.html
mod else_keyword {}

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
/// true, blah: "hello!".to_string(), }`. Empty Enums are similar to [`!`] in that they cannot be
/// instantiated at all, and are used mainly to mess with the type system in interesting ways.
///
/// For more information, take a look at the [Rust Book] or the [Reference]
///
/// [ADT]: https://en.wikipedia.org/wiki/Algebraic_data_type
/// [Rust Book]: ../book/ch06-01-defining-an-enum.html
/// [Reference]: ../reference/items/enumerations.html
mod enum_keyword {}

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
/// pub extern "C" fn callable_from_c(x: i32) -> bool {
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
/// [`crate`]: keyword.crate.html
mod extern_keyword {}

#[doc(keyword = "false")]
//
/// A value of type [`bool`] representing logical **false**.
///
/// `false` is the logical opposite of [`true`].
///
/// See the documentation for [`true`] for more information.
///
/// [`true`]: keyword.true.html
mod false_keyword {}

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
mod fn_keyword {}

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
/// practice within Rust, which is to loop over anything that implements [`IntoIterator`] until the
/// iterator returned by `.into_iter()` returns `None` (or the loop body uses `break`).
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
///     let result = match IntoIterator::into_iter(iterator) {
///         mut iter => loop {
///             let next;
///             match iter.next() {
///                 Some(val) => next = val,
///                 None => break,
///             };
///             let loop_variable = next;
///             let () = { code(); };
///         },
///     };
///     result
/// }
/// ```
///
/// More details on the functionality shown can be seen at the [`IntoIterator`] docs.
///
/// For more information on for-loops, see the [Rust book] or the [Reference].
///
/// See also, [`loop`], [`while`].
///
/// [`in`]: keyword.in.html
/// [`impl`]: keyword.impl.html
/// [`loop`]: keyword.loop.html
/// [`while`]: keyword.while.html
/// [higher-ranked trait bounds]: ../reference/trait-bounds.html#higher-ranked-trait-bounds
/// [Rust book]:
/// ../book/ch03-05-control-flow.html#looping-through-a-collection-with-for
/// [Reference]: ../reference/expressions/loop-expr.html#iterator-loops
mod for_keyword {}

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
mod if_keyword {}

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
mod impl_keyword {}

#[doc(keyword = "in")]
//
/// Iterate over a series of values with [`for`].
///
/// The expression immediately following `in` must implement the [`IntoIterator`] trait.
///
/// ## Literal Examples:
///
///    * `for _ in 1..3 {}` - Iterate over an exclusive range up to but excluding 3.
///    * `for _ in 1..=3 {}` - Iterate over an inclusive range up to and including 3.
///
/// (Read more about [range patterns])
///
/// [`IntoIterator`]: ../book/ch13-04-performance.html
/// [range patterns]: ../reference/patterns.html?highlight=range#range-patterns
/// [`for`]: keyword.for.html
mod in_keyword {}

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
mod let_keyword {}

#[doc(keyword = "while")]
//
/// Loop while a condition is upheld.
///
/// A `while` expression is used for predicate loops. The `while` expression runs the conditional
/// expression before running the loop body, then runs the loop body if the conditional
/// expression evaluates to `true`, or exits the loop otherwise.
///
/// ```rust
/// let mut counter = 0;
///
/// while counter < 10 {
///     println!("{}", counter);
///     counter += 1;
/// }
/// ```
///
/// Like the [`for`] expression, we can use `break` and `continue`. A `while` expression
/// cannot break with a value and always evaluates to `()` unlike [`loop`].
///
/// ```rust
/// let mut i = 1;
///
/// while i < 100 {
///     i *= 2;
///     if i == 64 {
///         break; // Exit when `i` is 64.
///     }
/// }
/// ```
///
/// As `if` expressions have their pattern matching variant in `if let`, so too do `while`
/// expressions with `while let`. The `while let` expression matches the pattern against the
/// expression, then runs the loop body if pattern matching succeeds, or exits the loop otherwise.
/// We can use `break` and `continue` in `while let` expressions just like in `while`.
///
/// ```rust
/// let mut counter = Some(0);
///
/// while let Some(i) = counter {
///     if i == 10 {
///         counter = None;
///     } else {
///         println!("{}", i);
///         counter = Some (i + 1);
///     }
/// }
/// ```
///
/// For more information on `while` and loops in general, see the [reference].
///
/// See also, [`for`], [`loop`].
///
/// [`for`]: keyword.for.html
/// [`loop`]: keyword.loop.html
/// [reference]: ../reference/expressions/loop-expr.html#predicate-loops
mod while_keyword {}

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
/// let mut i = 1;
/// loop {
///     println!("i is {}", i);
///     if i > 100 {
///         break;
///     }
///     i *= 2;
/// }
/// assert_eq!(i, 128);
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
/// See also, [`for`], [`while`].
///
/// [`for`]: keyword.for.html
/// [`while`]: keyword.while.html
/// [Reference]: ../reference/expressions/loop-expr.html
mod loop_keyword {}

#[doc(keyword = "match")]
//
/// Control flow based on pattern matching.
///
/// `match` can be used to run code conditionally. Every pattern must
/// be handled exhaustively either explicitly or by using wildcards like
/// `_` in the `match`. Since `match` is an expression, values can also be
/// returned.
///
/// ```rust
/// let opt = Option::None::<usize>;
/// let x = match opt {
///     Some(int) => int,
///     None => 10,
/// };
/// assert_eq!(x, 10);
///
/// let a_number = Option::Some(10);
/// match a_number {
///     Some(x) if x <= 5 => println!("0 to 5 num = {}", x),
///     Some(x @ 6..=10) => println!("6 to 10 num = {}", x),
///     None => panic!(),
///     // all other numbers
///     _ => panic!(),
/// }
/// ```
///
/// `match` can be used to gain access to the inner members of an enum
/// and use them directly.
///
/// ```rust
/// enum Outer {
///     Double(Option<u8>, Option<String>),
///     Single(Option<u8>),
///     Empty
/// }
///
/// let get_inner = Outer::Double(None, Some(String::new()));
/// match get_inner {
///     Outer::Double(None, Some(st)) => println!("{}", st),
///     Outer::Single(opt) => println!("{:?}", opt),
///     _ => panic!(),
/// }
/// ```
///
/// For more information on `match` and matching in general, see the [Reference].
///
/// [Reference]: ../reference/expressions/match-expr.html
mod match_keyword {}

#[doc(keyword = "mod")]
//
/// Organize code into [modules].
///
/// Use `mod` to create new [modules] to encapsulate code, including other
/// modules:
///
/// ```
/// mod foo {
///     mod bar {
///         type MyType = (u8, u8);
///         fn baz() {}
///     }
/// }
/// ```
///
/// Like [`struct`]s and [`enum`]s, a module and its content are private by
/// default, unaccessible to code outside of the module.
///
/// To learn more about allowing access, see the documentation for the [`pub`]
/// keyword.
///
/// [`enum`]: keyword.enum.html
/// [`pub`]: keyword.pub.html
/// [`struct`]: keyword.struct.html
/// [modules]: ../reference/items/modules.html
mod mod_keyword {}

#[doc(keyword = "move")]
//
/// Capture a [closure]'s environment by value.
///
/// `move` converts any variables captured by reference or mutable reference
/// to owned by value variables.
///
/// ```rust
/// let capture = "hello";
/// let closure = move || {
///     println!("rust says {}", capture);
/// };
/// ```
///
/// Note: `move` closures may still implement [`Fn`] or [`FnMut`], even though
/// they capture variables by `move`. This is because the traits implemented by
/// a closure type are determined by *what* the closure does with captured
/// values, not *how* it captures them:
///
/// ```rust
/// fn create_fn() -> impl Fn() {
///     let text = "Fn".to_owned();
///
///     move || println!("This is a: {}", text)
/// }
///
/// let fn_plain = create_fn();
///
/// fn_plain();
/// ```
///
/// `move` is often used when [threads] are involved.
///
/// ```rust
/// let x = 5;
///
/// std::thread::spawn(move || {
///     println!("captured {} by value", x)
/// }).join().unwrap();
///
/// // x is no longer available
/// ```
///
/// `move` is also valid before an async block.
///
/// ```rust
/// let capture = "hello";
/// let block = async move {
///     println!("rust says {} from async block", capture);
/// };
/// ```
///
/// For more information on the `move` keyword, see the [closures][closure] section
/// of the Rust book or the [threads] section.
///
/// [closure]: ../book/ch13-01-closures.html
/// [threads]: ../book/ch16-01-threads.html#using-move-closures-with-threads
mod move_keyword {}

#[doc(keyword = "mut")]
//
/// A mutable variable, reference, or pointer.
///
/// `mut` can be used in several situations. The first is mutable variables,
/// which can be used anywhere you can bind a value to a variable name. Some
/// examples:
///
/// ```rust
/// // A mutable variable in the parameter list of a function.
/// fn foo(mut x: u8, y: u8) -> u8 {
///     x += y;
///     x
/// }
///
/// // Modifying a mutable variable.
/// # #[allow(unused_assignments)]
/// let mut a = 5;
/// a = 6;
///
/// assert_eq!(foo(3, 4), 7);
/// assert_eq!(a, 6);
/// ```
///
/// The second is mutable references. They can be created from `mut` variables
/// and must be unique: no other variables can have a mutable reference, nor a
/// shared reference.
///
/// ```rust
/// // Taking a mutable reference.
/// fn push_two(v: &mut Vec<u8>) {
///     v.push(2);
/// }
///
/// // A mutable reference cannot be taken to a non-mutable variable.
/// let mut v = vec![0, 1];
/// // Passing a mutable reference.
/// push_two(&mut v);
///
/// assert_eq!(v, vec![0, 1, 2]);
/// ```
///
/// ```rust,compile_fail,E0502
/// let mut v = vec![0, 1];
/// let mut_ref_v = &mut v;
/// ##[allow(unused)]
/// let ref_v = &v;
/// mut_ref_v.push(2);
/// ```
///
/// Mutable raw pointers work much like mutable references, with the added
/// possibility of not pointing to a valid object. The syntax is `*mut Type`.
///
/// More information on mutable references and pointers can be found in```
/// [Reference].
///
/// [Reference]: ../reference/types/pointer.html#mutable-references-mut
mod mut_keyword {}

#[doc(keyword = "pub")]
//
/// Make an item visible to others.
///
/// The keyword `pub` makes any module, function, or data structure accessible from inside
/// of external modules. The `pub` keyword may also be used in a `use` declaration to re-export
/// an identifier from a namespace.
///
/// For more information on the `pub` keyword, please see the visibility section
/// of the [reference] and for some examples, see [Rust by Example].
///
/// [reference]:../reference/visibility-and-privacy.html?highlight=pub#visibility-and-privacy
/// [Rust by Example]:../rust-by-example/mod/visibility.html
mod pub_keyword {}

#[doc(keyword = "ref")]
//
/// Bind by reference during pattern matching.
///
/// `ref` annotates pattern bindings to make them borrow rather than move.
/// It is **not** a part of the pattern as far as matching is concerned: it does
/// not affect *whether* a value is matched, only *how* it is matched.
///
/// By default, [`match`] statements consume all they can, which can sometimes
/// be a problem, when you don't really need the value to be moved and owned:
///
/// ```compile_fail,E0382
/// let maybe_name = Some(String::from("Alice"));
/// // The variable 'maybe_name' is consumed here ...
/// match maybe_name {
///     Some(n) => println!("Hello, {}", n),
///     _ => println!("Hello, world"),
/// }
/// // ... and is now unavailable.
/// println!("Hello again, {}", maybe_name.unwrap_or("world".into()));
/// ```
///
/// Using the `ref` keyword, the value is only borrowed, not moved, making it
/// available for use after the [`match`] statement:
///
/// ```
/// let maybe_name = Some(String::from("Alice"));
/// // Using `ref`, the value is borrowed, not moved ...
/// match maybe_name {
///     Some(ref n) => println!("Hello, {}", n),
///     _ => println!("Hello, world"),
/// }
/// // ... so it's available here!
/// println!("Hello again, {}", maybe_name.unwrap_or("world".into()));
/// ```
///
/// # `&` vs `ref`
///
/// - `&` denotes that your pattern expects a reference to an object. Hence `&`
/// is a part of said pattern: `&Foo` matches different objects than `Foo` does.
///
/// - `ref` indicates that you want a reference to an unpacked value. It is not
/// matched against: `Foo(ref foo)` matches the same objects as `Foo(foo)`.
///
/// See also the [Reference] for more information.
///
/// [`match`]: keyword.match.html
/// [Reference]: ../reference/patterns.html#identifier-patterns
mod ref_keyword {}

#[doc(keyword = "return")]
//
/// Return a value from a function.
///
/// A `return` marks the end of an execution path in a function:
///
/// ```
/// fn foo() -> i32 {
///     return 3;
/// }
/// assert_eq!(foo(), 3);
/// ```
///
/// `return` is not needed when the returned value is the last expression in the
/// function. In this case the `;` is omitted:
///
/// ```
/// fn foo() -> i32 {
///     3
/// }
/// assert_eq!(foo(), 3);
/// ```
///
/// `return` returns from the function immediately (an "early return"):
///
/// ```no_run
/// use std::fs::File;
/// use std::io::{Error, ErrorKind, Read, Result};
///
/// fn main() -> Result<()> {
///     let mut file = match File::open("foo.txt") {
///         Ok(f) => f,
///         Err(e) => return Err(e),
///     };
///
///     let mut contents = String::new();
///     let size = match file.read_to_string(&mut contents) {
///         Ok(s) => s,
///         Err(e) => return Err(e),
///     };
///
///     if contents.contains("impossible!") {
///         return Err(Error::new(ErrorKind::Other, "oh no!"));
///     }
///
///     if size > 9000 {
///         return Err(Error::new(ErrorKind::Other, "over 9000!"));
///     }
///
///     assert_eq!(contents, "Hello, world!");
///     Ok(())
/// }
/// ```
mod return_keyword {}

#[doc(keyword = "self")]
//
/// The receiver of a method, or the current module.
///
/// `self` is used in two situations: referencing the current module and marking
/// the receiver of a method.
///
/// In paths, `self` can be used to refer to the current module, either in a
/// [`use`] statement or in a path to access an element:
///
/// ```
/// # #![allow(unused_imports)]
/// use std::io::{self, Read};
/// ```
///
/// Is functionally the same as:
///
/// ```
/// # #![allow(unused_imports)]
/// use std::io;
/// use std::io::Read;
/// ```
///
/// Using `self` to access an element in the current module:
///
/// ```
/// # #![allow(dead_code)]
/// # fn main() {}
/// fn foo() {}
/// fn bar() {
///     self::foo()
/// }
/// ```
///
/// `self` as the current receiver for a method allows to omit the parameter
/// type most of the time. With the exception of this particularity, `self` is
/// used much like any other parameter:
///
/// ```
/// struct Foo(i32);
///
/// impl Foo {
///     // No `self`.
///     fn new() -> Self {
///         Self(0)
///     }
///
///     // Consuming `self`.
///     fn consume(self) -> Self {
///         Self(self.0 + 1)
///     }
///
///     // Borrowing `self`.
///     fn borrow(&self) -> &i32 {
///         &self.0
///     }
///
///     // Borrowing `self` mutably.
///     fn borrow_mut(&mut self) -> &mut i32 {
///         &mut self.0
///     }
/// }
///
/// // This method must be called with a `Type::` prefix.
/// let foo = Foo::new();
/// assert_eq!(foo.0, 0);
///
/// // Those two calls produces the same result.
/// let foo = Foo::consume(foo);
/// assert_eq!(foo.0, 1);
/// let foo = foo.consume();
/// assert_eq!(foo.0, 2);
///
/// // Borrowing is handled automatically with the second syntax.
/// let borrow_1 = Foo::borrow(&foo);
/// let borrow_2 = foo.borrow();
/// assert_eq!(borrow_1, borrow_2);
///
/// // Borrowing mutably is handled automatically too with the second syntax.
/// let mut foo = Foo::new();
/// *Foo::borrow_mut(&mut foo) += 1;
/// assert_eq!(foo.0, 1);
/// *foo.borrow_mut() += 1;
/// assert_eq!(foo.0, 2);
/// ```
///
/// Note that this automatic conversion when calling `foo.method()` is not
/// limited to the examples above. See the [Reference] for more information.
///
/// [`use`]: keyword.use.html
/// [Reference]: ../reference/items/associated-items.html#methods
mod self_keyword {}

// FIXME: Once rustdoc can handle URL conflicts on case insensitive file systems, we can remove the
// three next lines and put back: `#[doc(keyword = "Self")]`.
#[doc(alias = "Self")]
#[allow(rustc::existing_doc_keyword)]
#[doc(keyword = "SelfTy")]
//
/// The implementing type within a [`trait`] or [`impl`] block, or the current type within a type
/// definition.
///
/// Within a type definition:
///
/// ```
/// # #![allow(dead_code)]
/// struct Node {
///     elem: i32,
///     // `Self` is a `Node` here.
///     next: Option<Box<Self>>,
/// }
/// ```
///
/// In an [`impl`] block:
///
/// ```
/// struct Foo(i32);
///
/// impl Foo {
///     fn new() -> Self {
///         Self(0)
///     }
/// }
///
/// assert_eq!(Foo::new().0, Foo(0).0);
/// ```
///
/// Generic parameters are implicit with `Self`:
///
/// ```
/// # #![allow(dead_code)]
/// struct Wrap<T> {
///     elem: T,
/// }
///
/// impl<T> Wrap<T> {
///     fn new(elem: T) -> Self {
///         Self { elem }
///     }
/// }
/// ```
///
/// In a [`trait`] definition and related [`impl`] block:
///
/// ```
/// trait Example {
///     fn example() -> Self;
/// }
///
/// struct Foo(i32);
///
/// impl Example for Foo {
///     fn example() -> Self {
///         Self(42)
///     }
/// }
///
/// assert_eq!(Foo::example().0, Foo(42).0);
/// ```
///
/// [`impl`]: keyword.impl.html
/// [`trait`]: keyword.trait.html
mod self_upper_keyword {}

#[doc(keyword = "static")]
//
/// A static item is a value which is valid for the entire duration of your
/// program (a `'static` lifetime).
///
/// On the surface, `static` items seem very similar to [`const`]s: both contain
/// a value, both require type annotations and both can only be initialized with
/// constant functions and values. However, `static`s are notably different in
/// that they represent a location in memory. That means that you can have
/// references to `static` items and potentially even modify them, making them
/// essentially global variables.
///
/// Static items do not call [`drop`] at the end of the program.
///
/// There are two types of `static` items: those declared in association with
/// the [`mut`] keyword and those without.
///
/// Static items cannot be moved:
///
/// ```rust,compile_fail,E0507
/// static VEC: Vec<u32> = vec![];
///
/// fn move_vec(v: Vec<u32>) -> Vec<u32> {
///     v
/// }
///
/// // This line causes an error
/// move_vec(VEC);
/// ```
///
/// # Simple `static`s
///
/// Accessing non-[`mut`] `static` items is considered safe, but some
/// restrictions apply. Most notably, the type of a `static` value needs to
/// implement the [`Sync`] trait, ruling out interior mutability containers
/// like [`RefCell`]. See the [Reference] for more information.
///
/// ```rust
/// static FOO: [i32; 5] = [1, 2, 3, 4, 5];
///
/// let r1 = &FOO as *const _;
/// let r2 = &FOO as *const _;
/// // With a strictly read-only static, references will have the same address
/// assert_eq!(r1, r2);
/// // A static item can be used just like a variable in many cases
/// println!("{:?}", FOO);
/// ```
///
/// # Mutable `static`s
///
/// If a `static` item is declared with the [`mut`] keyword, then it is allowed
/// to be modified by the program. However, accessing mutable `static`s can
/// cause undefined behavior in a number of ways, for example due to data races
/// in a multithreaded context. As such, all accesses to mutable `static`s
/// require an [`unsafe`] block.
///
/// Despite their unsafety, mutable `static`s are necessary in many contexts:
/// they can be used to represent global state shared by the whole program or in
/// [`extern`] blocks to bind to variables from C libraries.
///
/// In an [`extern`] block:
///
/// ```rust,no_run
/// # #![allow(dead_code)]
/// extern "C" {
///     static mut ERROR_MESSAGE: *mut std::os::raw::c_char;
/// }
/// ```
///
/// Mutable `static`s, just like simple `static`s, have some restrictions that
/// apply to them. See the [Reference] for more information.
///
/// [`const`]: keyword.const.html
/// [`extern`]: keyword.extern.html
/// [`mut`]: keyword.mut.html
/// [`unsafe`]: keyword.unsafe.html
/// [`RefCell`]: cell::RefCell
/// [Reference]: ../reference/items/static-items.html
mod static_keyword {}

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
/// [`PhantomData`]: marker::PhantomData
/// [book]: ../book/ch05-01-defining-structs.html
/// [reference]: ../reference/items/structs.html
mod struct_keyword {}

#[doc(keyword = "super")]
//
/// The parent of the current [module].
///
/// ```rust
/// # #![allow(dead_code)]
/// # fn main() {}
/// mod a {
///     pub fn foo() {}
/// }
/// mod b {
///     pub fn foo() {
///         super::a::foo(); // call a's foo function
///     }
/// }
/// ```
///
/// It is also possible to use `super` multiple times: `super::super::foo`,
/// going up the ancestor chain.
///
/// See the [Reference] for more information.
///
/// [module]: ../reference/items/modules.html
/// [Reference]: ../reference/paths.html#super
mod super_keyword {}

#[doc(keyword = "trait")]
//
/// A common interface for a group of types.
///
/// A `trait` is like an interface that data types can implement. When a type
/// implements a trait it can be treated abstractly as that trait using generics
/// or trait objects.
///
/// Traits can be made up of three varieties of associated items:
///
/// - functions and methods
/// - types
/// - constants
///
/// Traits may also contain additional type parameters. Those type parameters
/// or the trait itself can be constrained by other traits.
///
/// Traits can serve as markers or carry other logical semantics that
/// aren't expressed through their items. When a type implements that
/// trait it is promising to uphold its contract. [`Send`] and [`Sync`] are two
/// such marker traits present in the standard library.
///
/// See the [Reference][Ref-Traits] for a lot more information on traits.
///
/// # Examples
///
/// Traits are declared using the `trait` keyword. Types can implement them
/// using [`impl`] `Trait` [`for`] `Type`:
///
/// ```rust
/// trait Zero {
///     const ZERO: Self;
///     fn is_zero(&self) -> bool;
/// }
///
/// impl Zero for i32 {
///     const ZERO: Self = 0;
///
///     fn is_zero(&self) -> bool {
///         *self == Self::ZERO
///     }
/// }
///
/// assert_eq!(i32::ZERO, 0);
/// assert!(i32::ZERO.is_zero());
/// assert!(!4.is_zero());
/// ```
///
/// With an associated type:
///
/// ```rust
/// trait Builder {
///     type Built;
///
///     fn build(&self) -> Self::Built;
/// }
/// ```
///
/// Traits can be generic, with constraints or without:
///
/// ```rust
/// trait MaybeFrom<T> {
///     fn maybe_from(value: T) -> Option<Self>
///     where
///         Self: Sized;
/// }
/// ```
///
/// Traits can build upon the requirements of other traits. In the example
/// below `Iterator` is a **supertrait** and `ThreeIterator` is a **subtrait**:
///
/// ```rust
/// trait ThreeIterator: std::iter::Iterator {
///     fn next_three(&mut self) -> Option<[Self::Item; 3]>;
/// }
/// ```
///
/// Traits can be used in functions, as parameters:
///
/// ```rust
/// # #![allow(dead_code)]
/// fn debug_iter<I: Iterator>(it: I) where I::Item: std::fmt::Debug {
///     for elem in it {
///         println!("{:#?}", elem);
///     }
/// }
///
/// // u8_len_1, u8_len_2 and u8_len_3 are equivalent
///
/// fn u8_len_1(val: impl Into<Vec<u8>>) -> usize {
///     val.into().len()
/// }
///
/// fn u8_len_2<T: Into<Vec<u8>>>(val: T) -> usize {
///     val.into().len()
/// }
///
/// fn u8_len_3<T>(val: T) -> usize
/// where
///     T: Into<Vec<u8>>,
/// {
///     val.into().len()
/// }
/// ```
///
/// Or as return types:
///
/// ```rust
/// # #![allow(dead_code)]
/// fn from_zero_to(v: u8) -> impl Iterator<Item = u8> {
///     (0..v).into_iter()
/// }
/// ```
///
/// The use of the [`impl`] keyword in this position allows the function writer
/// to hide the concrete type as an implementation detail which can change
/// without breaking user's code.
///
/// # Trait objects
///
/// A *trait object* is an opaque value of another type that implements a set of
/// traits. A trait object implements all specified traits as well as their
/// supertraits (if any).
///
/// The syntax is the following: `dyn BaseTrait + AutoTrait1 + ... AutoTraitN`.
/// Only one `BaseTrait` can be used so this will not compile:
///
/// ```rust,compile_fail,E0225
/// trait A {}
/// trait B {}
///
/// let _: Box<dyn A + B>;
/// ```
///
/// Neither will this, which is a syntax error:
///
/// ```rust,compile_fail
/// trait A {}
/// trait B {}
///
/// let _: Box<dyn A + dyn B>;
/// ```
///
/// On the other hand, this is correct:
///
/// ```rust
/// trait A {}
///
/// let _: Box<dyn A + Send + Sync>;
/// ```
///
/// The [Reference][Ref-Trait-Objects] has more information about trait objects,
/// their limitations and the differences between editions.
///
/// # Unsafe traits
///
/// Some traits may be unsafe to implement. Using the [`unsafe`] keyword in
/// front of the trait's declaration is used to mark this:
///
/// ```rust
/// unsafe trait UnsafeTrait {}
///
/// unsafe impl UnsafeTrait for i32 {}
/// ```
///
/// # Differences between the 2015 and 2018 editions
///
/// In the 2015 edition the parameters pattern was not needed for traits:
///
/// ```rust,edition2015
/// # #![allow(anonymous_parameters)]
/// trait Tr {
///     fn f(i32);
/// }
/// ```
///
/// This behavior is no longer valid in edition 2018.
///
/// [`for`]: keyword.for.html
/// [`impl`]: keyword.impl.html
/// [`unsafe`]: keyword.unsafe.html
/// [Ref-Traits]: ../reference/items/traits.html
/// [Ref-Trait-Objects]: ../reference/types/trait-object.html
mod trait_keyword {}

#[doc(keyword = "true")]
//
/// A value of type [`bool`] representing logical **true**.
///
/// Logically `true` is not equal to [`false`].
///
/// ## Control structures that check for **true**
///
/// Several of Rust's control structures will check for a `bool` condition evaluating to **true**.
///
///   * The condition in an [`if`] expression must be of type `bool`.
///     Whenever that condition evaluates to **true**, the `if` expression takes
///     on the value of the first block. If however, the condition evaluates
///     to `false`, the expression takes on value of the `else` block if there is one.
///
///   * [`while`] is another control flow construct expecting a `bool`-typed condition.
///     As long as the condition evaluates to **true**, the `while` loop will continually
///     evaluate its associated block.
///
///   * [`match`] arms can have guard clauses on them.
///
/// [`if`]: keyword.if.html
/// [`while`]: keyword.while.html
/// [`match`]: ../reference/expressions/match-expr.html#match-guards
/// [`false`]: keyword.false.html
mod true_keyword {}

#[doc(keyword = "type")]
//
/// Define an alias for an existing type.
///
/// The syntax is `type Name = ExistingType;`.
///
/// # Examples
///
/// `type` does **not** create a new type:
///
/// ```rust
/// type Meters = u32;
/// type Kilograms = u32;
///
/// let m: Meters = 3;
/// let k: Kilograms = 3;
///
/// assert_eq!(m, k);
/// ```
///
/// In traits, `type` is used to declare an [associated type]:
///
/// ```rust
/// trait Iterator {
///     // associated type declaration
///     type Item;
///     fn next(&mut self) -> Option<Self::Item>;
/// }
///
/// struct Once<T>(Option<T>);
///
/// impl<T> Iterator for Once<T> {
///     // associated type definition
///     type Item = T;
///     fn next(&mut self) -> Option<Self::Item> {
///         self.0.take()
///     }
/// }
/// ```
///
/// [`trait`]: keyword.trait.html
/// [associated type]: ../reference/items/associated-items.html#associated-types
mod type_keyword {}

#[doc(keyword = "unsafe")]
//
/// Code or interfaces whose [memory safety] cannot be verified by the type
/// system.
///
/// The `unsafe` keyword has two uses: to declare the existence of contracts the
/// compiler can't check (`unsafe fn` and `unsafe trait`), and to declare that a
/// programmer has checked that these contracts have been upheld (`unsafe {}`
/// and `unsafe impl`, but also `unsafe fn` -- see below). They are not mutually
/// exclusive, as can be seen in `unsafe fn`.
///
/// # Unsafe abilities
///
/// **No matter what, Safe Rust can't cause Undefined Behavior**. This is
/// referred to as [soundness]: a well-typed program actually has the desired
/// properties. The [Nomicon][nomicon-soundness] has a more detailed explanation
/// on the subject.
///
/// To ensure soundness, Safe Rust is restricted enough that it can be
/// automatically checked. Sometimes, however, it is necessary to write code
/// that is correct for reasons which are too clever for the compiler to
/// understand. In those cases, you need to use Unsafe Rust.
///
/// Here are the abilities Unsafe Rust has in addition to Safe Rust:
///
/// - Dereference [raw pointers]
/// - Implement `unsafe` [`trait`]s
/// - Call `unsafe` functions
/// - Mutate [`static`]s (including [`extern`]al ones)
/// - Access fields of [`union`]s
///
/// However, this extra power comes with extra responsibilities: it is now up to
/// you to ensure soundness. The `unsafe` keyword helps by clearly marking the
/// pieces of code that need to worry about this.
///
/// ## The different meanings of `unsafe`
///
/// Not all uses of `unsafe` are equivalent: some are here to mark the existence
/// of a contract the programmer must check, others are to say "I have checked
/// the contract, go ahead and do this". The following
/// [discussion on Rust Internals] has more in-depth explanations about this but
/// here is a summary of the main points:
///
/// - `unsafe fn`: calling this function means abiding by a contract the
/// compiler cannot enforce.
/// - `unsafe trait`: implementing the [`trait`] means abiding by a
/// contract the compiler cannot enforce.
/// - `unsafe {}`: the contract necessary to call the operations inside the
/// block has been checked by the programmer and is guaranteed to be respected.
/// - `unsafe impl`: the contract necessary to implement the trait has been
/// checked by the programmer and is guaranteed to be respected.
///
/// `unsafe fn` also acts like an `unsafe {}` block
/// around the code inside the function. This means it is not just a signal to
/// the caller, but also promises that the preconditions for the operations
/// inside the function are upheld. Mixing these two meanings can be confusing
/// and [proposal]s exist to use `unsafe {}` blocks inside such functions when
/// making `unsafe` operations.
///
/// See the [Rustnomicon] and the [Reference] for more informations.
///
/// # Examples
///
/// ## Marking elements as `unsafe`
///
/// `unsafe` can be used on functions. Note that functions and statics declared
/// in [`extern`] blocks are implicitly marked as `unsafe` (but not functions
/// declared as `extern "something" fn ...`). Mutable statics are always unsafe,
/// wherever they are declared. Methods can also be declared as `unsafe`:
///
/// ```rust
/// # #![allow(dead_code)]
/// static mut FOO: &str = "hello";
///
/// unsafe fn unsafe_fn() {}
///
/// extern "C" {
///     fn unsafe_extern_fn();
///     static BAR: *mut u32;
/// }
///
/// trait SafeTraitWithUnsafeMethod {
///     unsafe fn unsafe_method(&self);
/// }
///
/// struct S;
///
/// impl S {
///     unsafe fn unsafe_method_on_struct() {}
/// }
/// ```
///
/// Traits can also be declared as `unsafe`:
///
/// ```rust
/// unsafe trait UnsafeTrait {}
/// ```
///
/// Since `unsafe fn` and `unsafe trait` indicate that there is a safety
/// contract that the compiler cannot enforce, documenting it is important. The
/// standard library has many examples of this, like the following which is an
/// extract from [`Vec::set_len`]. The `# Safety` section explains the contract
/// that must be fulfilled to safely call the function.
///
/// ```rust,ignore (stub-to-show-doc-example)
/// /// Forces the length of the vector to `new_len`.
/// ///
/// /// This is a low-level operation that maintains none of the normal
/// /// invariants of the type. Normally changing the length of a vector
/// /// is done using one of the safe operations instead, such as
/// /// `truncate`, `resize`, `extend`, or `clear`.
/// ///
/// /// # Safety
/// ///
/// /// - `new_len` must be less than or equal to `capacity()`.
/// /// - The elements at `old_len..new_len` must be initialized.
/// pub unsafe fn set_len(&mut self, new_len: usize)
/// ```
///
/// ## Using `unsafe {}` blocks and `impl`s
///
/// Performing `unsafe` operations requires an `unsafe {}` block:
///
/// ```rust
/// # #![allow(dead_code)]
/// /// Dereference the given pointer.
/// ///
/// /// # Safety
/// ///
/// /// `ptr` must be aligned and must not be dangling.
/// unsafe fn deref_unchecked(ptr: *const i32) -> i32 {
///     *ptr
/// }
///
/// let a = 3;
/// let b = &a as *const _;
/// // SAFETY: `a` has not been dropped and references are always aligned,
/// // so `b` is a valid address.
/// unsafe { assert_eq!(*b, deref_unchecked(b)); };
/// ```
///
/// Traits marked as `unsafe` must be [`impl`]emented using `unsafe impl`. This
/// makes a guarantee to other `unsafe` code that the implementation satisfies
/// the trait's safety contract. The [Send] and [Sync] traits are examples of
/// this behaviour in the standard library.
///
/// ```rust
/// /// Implementors of this trait must guarantee an element is always
/// /// accessible with index 3.
/// unsafe trait ThreeIndexable<T> {
///     /// Returns a reference to the element with index 3 in `&self`.
///     fn three(&self) -> &T;
/// }
///
/// // The implementation of `ThreeIndexable` for `[T; 4]` is `unsafe`
/// // because the implementor must abide by a contract the compiler cannot
/// // check but as a programmer we know there will always be a valid element
/// // at index 3 to access.
/// unsafe impl<T> ThreeIndexable<T> for [T; 4] {
///     fn three(&self) -> &T {
///         // SAFETY: implementing the trait means there always is an element
///         // with index 3 accessible.
///         unsafe { self.get_unchecked(3) }
///     }
/// }
///
/// let a = [1, 2, 4, 8];
/// assert_eq!(a.three(), &8);
/// ```
///
/// [`extern`]: keyword.extern.html
/// [`trait`]: keyword.trait.html
/// [`static`]: keyword.static.html
/// [`union`]: keyword.union.html
/// [`impl`]: keyword.impl.html
/// [raw pointers]: ../reference/types/pointer.html
/// [memory safety]: ../book/ch19-01-unsafe-rust.html
/// [Rustnomicon]: ../nomicon/index.html
/// [nomicon-soundness]: ../nomicon/safe-unsafe-meaning.html
/// [soundness]: https://rust-lang.github.io/unsafe-code-guidelines/glossary.html#soundness-of-code--of-a-library
/// [Reference]: ../reference/unsafety.html
/// [proposal]: https://github.com/rust-lang/rfcs/pull/2585
/// [discussion on Rust Internals]: https://internals.rust-lang.org/t/what-does-unsafe-mean/6696
mod unsafe_keyword {}

#[doc(keyword = "use")]
//
/// Import or rename items from other crates or modules.
///
/// Usually a `use` keyword is used to shorten the path required to refer to a module item.
/// The keyword may appear in modules, blocks and even functions, usually at the top.
///
/// The most basic usage of the keyword is `use path::to::item;`,
/// though a number of convenient shortcuts are supported:
///
///   * Simultaneously binding a list of paths with a common prefix,
///     using the glob-like brace syntax `use a::b::{c, d, e::f, g::h::i};`
///   * Simultaneously binding a list of paths with a common prefix and their common parent module,
///     using the [`self`] keyword, such as `use a::b::{self, c, d::e};`
///   * Rebinding the target name as a new local name, using the syntax `use p::q::r as x;`.
///     This can also be used with the last two features: `use a::b::{self as ab, c as abc}`.
///   * Binding all paths matching a given prefix,
///     using the asterisk wildcard syntax `use a::b::*;`.
///   * Nesting groups of the previous features multiple times,
///     such as `use a::b::{self as ab, c, d::{*, e::f}};`
///   * Reexporting with visibility modifiers such as `pub use a::b;`
///   * Importing with `_` to only import the methods of a trait without binding it to a name
///     (to avoid conflict for example): `use ::std::io::Read as _;`.
///
/// Using path qualifiers like [`crate`], [`super`] or [`self`] is supported: `use crate::a::b;`.
///
/// Note that when the wildcard `*` is used on a type, it does not import its methods (though
/// for `enum`s it imports the variants, as shown in the example below).
///
/// ```compile_fail,edition2018
/// enum ExampleEnum {
///     VariantA,
///     VariantB,
/// }
///
/// impl ExampleEnum {
///     fn new() -> Self {
///         Self::VariantA
///     }
/// }
///
/// use ExampleEnum::*;
///
/// // Compiles.
/// let _ = VariantA;
///
/// // Does not compile !
/// let n = new();
/// ```
///
/// For more information on `use` and paths in general, see the [Reference].
///
/// The differences about paths and the `use` keyword between the 2015 and 2018 editions
/// can also be found in the [Reference].
///
/// [`crate`]: keyword.crate.html
/// [`self`]: keyword.self.html
/// [`super`]: keyword.super.html
/// [Reference]: ../reference/items/use-declarations.html
mod use_keyword {}

#[doc(keyword = "where")]
//
/// Add constraints that must be upheld to use an item.
///
/// `where` allows specifying constraints on lifetime and generic parameters.
/// The [RFC] introducing `where` contains detailed informations about the
/// keyword.
///
/// # Examples
///
/// `where` can be used for constraints with traits:
///
/// ```rust
/// fn new<T: Default>() -> T {
///     T::default()
/// }
///
/// fn new_where<T>() -> T
/// where
///     T: Default,
/// {
///     T::default()
/// }
///
/// assert_eq!(0.0, new());
/// assert_eq!(0.0, new_where());
///
/// assert_eq!(0, new());
/// assert_eq!(0, new_where());
/// ```
///
/// `where` can also be used for lifetimes.
///
/// This compiles because `longer` outlives `shorter`, thus the constraint is
/// respected:
///
/// ```rust
/// fn select<'short, 'long>(s1: &'short str, s2: &'long str, second: bool) -> &'short str
/// where
///     'long: 'short,
/// {
///     if second { s2 } else { s1 }
/// }
///
/// let outer = String::from("Long living ref");
/// let longer = &outer;
/// {
///     let inner = String::from("Short living ref");
///     let shorter = &inner;
///
///     assert_eq!(select(shorter, longer, false), shorter);
///     assert_eq!(select(shorter, longer, true), longer);
/// }
/// ```
///
/// On the other hand, this will not compile because the `where 'b: 'a` clause
/// is missing: the `'b` lifetime is not known to live at least as long as `'a`
/// which means this function cannot ensure it always returns a valid reference:
///
/// ```rust,compile_fail,E0623
/// fn select<'a, 'b>(s1: &'a str, s2: &'b str, second: bool) -> &'a str
/// {
///     if second { s2 } else { s1 }
/// }
/// ```
///
/// `where` can also be used to express more complicated constraints that cannot
/// be written with the `<T: Trait>` syntax:
///
/// ```rust
/// fn first_or_default<I>(mut i: I) -> I::Item
/// where
///     I: Iterator,
///     I::Item: Default,
/// {
///     i.next().unwrap_or_else(I::Item::default)
/// }
///
/// assert_eq!(first_or_default(vec![1, 2, 3].into_iter()), 1);
/// assert_eq!(first_or_default(Vec::<i32>::new().into_iter()), 0);
/// ```
///
/// `where` is available anywhere generic and lifetime parameters are available,
/// as can be seen with the [`Cow`](crate::borrow::Cow) type from the standard
/// library:
///
/// ```rust
/// # #![allow(dead_code)]
/// pub enum Cow<'a, B>
/// where
///     B: 'a + ToOwned + ?Sized,
///  {
///     Borrowed(&'a B),
///     Owned(<B as ToOwned>::Owned),
/// }
/// ```
///
/// [RFC]: https://github.com/rust-lang/rfcs/blob/master/text/0135-where.md
mod where_keyword {}

// 2018 Edition keywords

#[doc(alias = "promise")]
#[doc(keyword = "async")]
//
/// Return a [`Future`] instead of blocking the current thread.
///
/// Use `async` in front of `fn`, `closure`, or a `block` to turn the marked code into a `Future`.
/// As such the code will not be run immediately, but will only be evaluated when the returned
/// future is `.await`ed.
///
/// We have written an [async book] detailing async/await and trade-offs compared to using threads.
///
/// ## Editions
///
/// `async` is a keyword from the 2018 edition onwards.
///
/// It is available for use in stable rust from version 1.39 onwards.
///
/// [`Future`]: future::Future
/// [async book]: https://rust-lang.github.io/async-book/
mod async_keyword {}

#[doc(keyword = "await")]
//
/// Suspend execution until the result of a [`Future`] is ready.
///
/// `.await`ing a future will suspend the current function's execution until the `executor`
/// has run the future to completion.
///
/// Read the [async book] for details on how async/await and executors work.
///
/// ## Editions
///
/// `await` is a keyword from the 2018 edition onwards.
///
/// It is available for use in stable rust from version 1.39 onwards.
///
/// [`Future`]: future::Future
/// [async book]: https://rust-lang.github.io/async-book/
mod await_keyword {}

#[doc(keyword = "dyn")]
//
/// `dyn` is a prefix of a [trait object]'s type.
///
/// The `dyn` keyword is used to highlight that calls to methods on the associated `Trait`
/// are dynamically dispatched. To use the trait this way, it must be 'object safe'.
///
/// Unlike generic parameters or `impl Trait`, the compiler does not know the concrete type that
/// is being passed. That is, the type has been [erased].
/// As such, a `dyn Trait` reference contains _two_ pointers.
/// One pointer goes to the data (e.g., an instance of a struct).
/// Another pointer goes to a map of method call names to function pointers
/// (known as a virtual method table or vtable).
///
/// At run-time, when a method needs to be called on the `dyn Trait`, the vtable is consulted to get
/// the function pointer and then that function pointer is called.
///
/// See the Reference for more information on [trait objects][ref-trait-obj]
/// and [object safety][ref-obj-safety].
///
/// ## Trade-offs
///
/// The above indirection is the additional runtime cost of calling a function on a `dyn Trait`.
/// Methods called by dynamic dispatch generally cannot be inlined by the compiler.
///
/// However, `dyn Trait` is likely to produce smaller code than `impl Trait` / generic parameters as
/// the method won't be duplicated for each concrete type.
///
/// [trait object]: ../book/ch17-02-trait-objects.html
/// [ref-trait-obj]: ../reference/types/trait-object.html
/// [ref-obj-safety]: ../reference/items/traits.html#object-safety
/// [erased]: https://en.wikipedia.org/wiki/Type_erasure
mod dyn_keyword {}

#[doc(keyword = "union")]
//
/// The [Rust equivalent of a C-style union][union].
///
/// A `union` looks like a [`struct`] in terms of declaration, but all of its
/// fields exist in the same memory, superimposed over one another. For instance,
/// if we wanted some bits in memory that we sometimes interpret as a `u32` and
/// sometimes as an `f32`, we could write:
///
/// ```rust
/// union IntOrFloat {
///     i: u32,
///     f: f32,
/// }
///
/// let mut u = IntOrFloat { f: 1.0 };
/// // Reading the fields of an union is always unsafe
/// assert_eq!(unsafe { u.i }, 1065353216);
/// // Updating through any of the field will modify all of them
/// u.i = 1073741824;
/// assert_eq!(unsafe { u.f }, 2.0);
/// ```
///
/// # Matching on unions
///
/// It is possible to use pattern matching on `union`s. A single field name must
/// be used and it must match the name of one of the `union`'s field.
/// Like reading from a `union`, pattern matching on a `union` requires `unsafe`.
///
/// ```rust
/// union IntOrFloat {
///     i: u32,
///     f: f32,
/// }
///
/// let u = IntOrFloat { f: 1.0 };
///
/// unsafe {
///     match u {
///         IntOrFloat { i: 10 } => println!("Found exactly ten!"),
///         // Matching the field `f` provides an `f32`.
///         IntOrFloat { f } => println!("Found f = {} !", f),
///     }
/// }
/// ```
///
/// # References to union fields
///
/// All fields in a `union` are all at the same place in memory which means
/// borrowing one borrows the entire `union`, for the same lifetime:
///
/// ```rust,compile_fail,E0502
/// union IntOrFloat {
///     i: u32,
///     f: f32,
/// }
///
/// let mut u = IntOrFloat { f: 1.0 };
///
/// let f = unsafe { &u.f };
/// // This will not compile because the field has already been borrowed, even
/// // if only immutably
/// let i = unsafe { &mut u.i };
///
/// *i = 10;
/// println!("f = {} and i = {}", f, i);
/// ```
///
/// See the [Reference][union] for more informations on `union`s.
///
/// [`struct`]: keyword.struct.html
/// [union]: ../reference/items/unions.html
mod union_keyword {}
