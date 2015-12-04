// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

// Error messages for EXXXX errors.
// Each message should start and end with a new line, and be wrapped to 80 characters.
// In vim you can `:set tw=80` and use `gq` to wrap paragraphs. Use `:set tw=0` to disable.
register_long_diagnostics! {

E0001: r##"
This error suggests that the expression arm corresponding to the noted pattern
will never be reached as for all possible values of the expression being
matched, one of the preceding patterns will match.

This means that perhaps some of the preceding patterns are too general, this one
is too specific or the ordering is incorrect.

For example, the following `match` block has too many arms:

```
match foo {
    Some(bar) => {/* ... */}
    None => {/* ... */}
    _ => {/* ... */} // All possible cases have already been handled
}
```

`match` blocks have their patterns matched in order, so, for example, putting
a wildcard arm above a more specific arm will make the latter arm irrelevant.

Ensure the ordering of the match arm is correct and remove any superfluous
arms.
"##,

E0002: r##"
This error indicates that an empty match expression is invalid because the type
it is matching on is non-empty (there exist values of this type). In safe code
it is impossible to create an instance of an empty type, so empty match
expressions are almost never desired. This error is typically fixed by adding
one or more cases to the match expression.

An example of an empty type is `enum Empty { }`. So, the following will work:

```
fn foo(x: Empty) {
    match x {
        // empty
    }
}
```

However, this won't:

```
fn foo(x: Option<String>) {
    match x {
        // empty
    }
}
```
"##,

E0003: r##"
Not-a-Number (NaN) values cannot be compared for equality and hence can never
match the input to a match expression. So, the following will not compile:

```
const NAN: f32 = 0.0 / 0.0;

match number {
    NAN => { /* ... */ },
    // ...
}
```

To match against NaN values, you should instead use the `is_nan()` method in a
guard, like so:

```
match number {
    // ...
    x if x.is_nan() => { /* ... */ }
    // ...
}
```
"##,

E0004: r##"
This error indicates that the compiler cannot guarantee a matching pattern for
one or more possible inputs to a match expression. Guaranteed matches are
required in order to assign values to match expressions, or alternatively,
determine the flow of execution.

If you encounter this error you must alter your patterns so that every possible
value of the input type is matched. For types with a small number of variants
(like enums) you should probably cover all cases explicitly. Alternatively, the
underscore `_` wildcard pattern can be added after all other patterns to match
"anything else".
"##,

E0005: r##"
Patterns used to bind names must be irrefutable, that is, they must guarantee
that a name will be extracted in all cases. If you encounter this error you
probably need to use a `match` or `if let` to deal with the possibility of
failure.
"##,

E0007: r##"
This error indicates that the bindings in a match arm would require a value to
be moved into more than one location, thus violating unique ownership. Code like
the following is invalid as it requires the entire `Option<String>` to be moved
into a variable called `op_string` while simultaneously requiring the inner
String to be moved into a variable called `s`.

```
let x = Some("s".to_string());
match x {
    op_string @ Some(s) => ...
    None => ...
}
```

See also Error 303.
"##,

E0008: r##"
Names bound in match arms retain their type in pattern guards. As such, if a
name is bound by move in a pattern, it should also be moved to wherever it is
referenced in the pattern guard code. Doing so however would prevent the name
from being available in the body of the match arm. Consider the following:

```
match Some("hi".to_string()) {
    Some(s) if s.len() == 0 => // use s.
    ...
}
```

The variable `s` has type `String`, and its use in the guard is as a variable of
type `String`. The guard code effectively executes in a separate scope to the
body of the arm, so the value would be moved into this anonymous scope and
therefore become unavailable in the body of the arm. Although this example seems
innocuous, the problem is most clear when considering functions that take their
argument by value.

```
match Some("hi".to_string()) {
    Some(s) if { drop(s); false } => (),
    Some(s) => // use s.
    ...
}
```

The value would be dropped in the guard then become unavailable not only in the
body of that arm but also in all subsequent arms! The solution is to bind by
reference when using guards or refactor the entire expression, perhaps by
putting the condition inside the body of the arm.
"##,

E0009: r##"
In a pattern, all values that don't implement the `Copy` trait have to be bound
the same way. The goal here is to avoid binding simultaneously by-move and
by-ref.

This limitation may be removed in a future version of Rust.

Wrong example:

```
struct X { x: (), }

let x = Some((X { x: () }, X { x: () }));
match x {
    Some((y, ref z)) => {},
    None => panic!()
}
```

You have two solutions:

Solution #1: Bind the pattern's values the same way.

```
struct X { x: (), }

let x = Some((X { x: () }, X { x: () }));
match x {
    Some((ref y, ref z)) => {},
    // or Some((y, z)) => {}
    None => panic!()
}
```

Solution #2: Implement the `Copy` trait for the `X` structure.

However, please keep in mind that the first solution should be preferred.

```
#[derive(Clone, Copy)]
struct X { x: (), }

let x = Some((X { x: () }, X { x: () }));
match x {
    Some((y, ref z)) => {},
    None => panic!()
}
```
"##,

E0010: r##"
The value of statics and constants must be known at compile time, and they live
for the entire lifetime of a program. Creating a boxed value allocates memory on
the heap at runtime, and therefore cannot be done at compile time. Erroneous
code example:

```
#![feature(box_syntax)]

const CON : Box<i32> = box 0;
```
"##,

E0011: r##"
Initializers for constants and statics are evaluated at compile time.
User-defined operators rely on user-defined functions, which cannot be evaluated
at compile time.

Bad example:

```
use std::ops::Index;

struct Foo { a: u8 }

impl Index<u8> for Foo {
    type Output = u8;

    fn index<'a>(&'a self, idx: u8) -> &'a u8 { &self.a }
}

const a: Foo = Foo { a: 0u8 };
const b: u8 = a[0]; // Index trait is defined by the user, bad!
```

Only operators on builtin types are allowed.

Example:

```
const a: &'static [i32] = &[1, 2, 3];
const b: i32 = a[0]; // Good!
```
"##,

E0013: r##"
Static and const variables can refer to other const variables. But a const
variable cannot refer to a static variable. For example, `Y` cannot refer to `X`
here:

```
static X: i32 = 42;
const Y: i32 = X;
```

To fix this, the value can be extracted as a const and then used:

```
const A: i32 = 42;
static X: i32 = A;
const Y: i32 = A;
```
"##,

E0014: r##"
Constants can only be initialized by a constant value or, in a future
version of Rust, a call to a const function. This error indicates the use
of a path (like a::b, or x) denoting something other than one of these
allowed items. Example:

```
const FOO: i32 = { let x = 0; x }; // 'x' isn't a constant nor a function!
```

To avoid it, you have to replace the non-constant value:

```
const FOO: i32 = { const X : i32 = 0; X };
// or even:
const FOO: i32 = { 0 }; // but brackets are useless here
```
"##,

// FIXME(#24111) Change the language here when const fn stabilizes
E0015: r##"
The only functions that can be called in static or constant expressions are
`const` functions, and struct/enum constructors. `const` functions are only
available on a nightly compiler. Rust currently does not support more general
compile-time function execution.

```
const FOO: Option<u8> = Some(1); // enum constructor
struct Bar {x: u8}
const BAR: Bar = Bar {x: 1}; // struct constructor
```

See [RFC 911] for more details on the design of `const fn`s.

[RFC 911]: https://github.com/rust-lang/rfcs/blob/master/text/0911-const-fn.md
"##,

E0016: r##"
Blocks in constants may only contain items (such as constant, function
definition, etc...) and a tail expression. Example:

```
const FOO: i32 = { let x = 0; x }; // 'x' isn't an item!
```

To avoid it, you have to replace the non-item object:

```
const FOO: i32 = { const X : i32 = 0; X };
```
"##,

E0017: r##"
References in statics and constants may only refer to immutable values. Example:

```
static X: i32 = 1;
const C: i32 = 2;

// these three are not allowed:
const CR: &'static mut i32 = &mut C;
static STATIC_REF: &'static mut i32 = &mut X;
static CONST_REF: &'static mut i32 = &mut C;
```

Statics are shared everywhere, and if they refer to mutable data one might
violate memory safety since holding multiple mutable references to shared data
is not allowed.

If you really want global mutable state, try using `static mut` or a global
`UnsafeCell`.
"##,

E0018: r##"
The value of static and const variables must be known at compile time. You
can't cast a pointer as an integer because we can't know what value the
address will take.

However, pointers to other constants' addresses are allowed in constants,
example:

```
const X: u32 = 50;
const Y: *const u32 = &X;
```

Therefore, casting one of these non-constant pointers to an integer results
in a non-constant integer which lead to this error. Example:

```
const X: u32 = 1;
const Y: usize = &X as *const u32 as usize;
println!("{}", Y);
```
"##,

E0019: r##"
A function call isn't allowed in the const's initialization expression
because the expression's value must be known at compile-time. Example of
erroneous code:

```
enum Test {
    V1
}

impl Test {
    fn test(&self) -> i32 {
        12
    }
}

fn main() {
    const FOO: Test = Test::V1;

    const A: i32 = FOO.test(); // You can't call Test::func() here !
}
```

Remember: you can't use a function call inside a const's initialization
expression! However, you can totally use it anywhere else:

```
fn main() {
    const FOO: Test = Test::V1;

    FOO.func(); // here is good
    let x = FOO.func(); // or even here!
}
```
"##,

E0020: r##"
This error indicates that an attempt was made to divide by zero (or take the
remainder of a zero divisor) in a static or constant expression. Erroneous
code example:

```
const X: i32 = 42 / 0;
// error: attempted to divide by zero in a constant expression
```
"##,

E0022: r##"
Constant functions are not allowed to mutate anything. Thus, binding to an
argument with a mutable pattern is not allowed. For example,

```
const fn foo(mut x: u8) {
    // do stuff
}
```

is bad because the function body may not mutate `x`.

Remove any mutable bindings from the argument list to fix this error. In case
you need to mutate the argument, try lazily initializing a global variable
instead of using a `const fn`, or refactoring the code to a functional style to
avoid mutation if possible.
"##,

E0030: r##"
When matching against a range, the compiler verifies that the range is
non-empty.  Range patterns include both end-points, so this is equivalent to
requiring the start of the range to be less than or equal to the end of the
range.

For example:

```
match 5u32 {
    // This range is ok, albeit pointless.
    1 ... 1 => ...
    // This range is empty, and the compiler can tell.
    1000 ... 5 => ...
}
```
"##,

E0038: r####"
Trait objects like `Box<Trait>` can only be constructed when certain
requirements are satisfied by the trait in question.

Trait objects are a form of dynamic dispatch and use a dynamically sized type
for the inner type. So, for a given trait `Trait`, when `Trait` is treated as a
type, as in `Box<Trait>`, the inner type is 'unsized'. In such cases the boxed
pointer is a 'fat pointer' that contains an extra pointer to a table of methods
(among other things) for dynamic dispatch. This design mandates some
restrictions on the types of traits that are allowed to be used in trait
objects, which are collectively termed as 'object safety' rules.

Attempting to create a trait object for a non object-safe trait will trigger
this error.

There are various rules:

### The trait cannot require `Self: Sized`

When `Trait` is treated as a type, the type does not implement the special
`Sized` trait, because the type does not have a known size at compile time and
can only be accessed behind a pointer. Thus, if we have a trait like the
following:

```
trait Foo where Self: Sized {

}
```

we cannot create an object of type `Box<Foo>` or `&Foo` since in this case
`Self` would not be `Sized`.

Generally, `Self : Sized` is used to indicate that the trait should not be used
as a trait object. If the trait comes from your own crate, consider removing
this restriction.

### Method references the `Self` type in its arguments or return type

This happens when a trait has a method like the following:

```
trait Trait {
    fn foo(&self) -> Self;
}

impl Trait for String {
    fn foo(&self) -> Self {
        "hi".to_owned()
    }
}

impl Trait for u8 {
    fn foo(&self) -> Self {
        1
    }
}
```

(Note that `&self` and `&mut self` are okay, it's additional `Self` types which
cause this problem)

In such a case, the compiler cannot predict the return type of `foo()` in a
situation like the following:

```
fn call_foo(x: Box<Trait>) {
    let y = x.foo(); // What type is y?
    // ...
}
```

If only some methods aren't object-safe, you can add a `where Self: Sized` bound
on them to mark them as explicitly unavailable to trait objects. The
functionality will still be available to all other implementers, including
`Box<Trait>` which is itself sized (assuming you `impl Trait for Box<Trait>`).

```
trait Trait {
    fn foo(&self) -> Self where Self: Sized;
    // more functions
}
```

Now, `foo()` can no longer be called on a trait object, but you will now be
allowed to make a trait object, and that will be able to call any object-safe
methods". With such a bound, one can still call `foo()` on types implementing
that trait that aren't behind trait objects.

### Method has generic type parameters

As mentioned before, trait objects contain pointers to method tables. So, if we
have:

```
trait Trait {
    fn foo(&self);
}
impl Trait for String {
    fn foo(&self) {
        // implementation 1
    }
}
impl Trait for u8 {
    fn foo(&self) {
        // implementation 2
    }
}
// ...
```

At compile time each implementation of `Trait` will produce a table containing
the various methods (and other items) related to the implementation.

This works fine, but when the method gains generic parameters, we can have a
problem.

Usually, generic parameters get _monomorphized_. For example, if I have

```
fn foo<T>(x: T) {
    // ...
}
```

the machine code for `foo::<u8>()`, `foo::<bool>()`, `foo::<String>()`, or any
other type substitution is different. Hence the compiler generates the
implementation on-demand. If you call `foo()` with a `bool` parameter, the
compiler will only generate code for `foo::<bool>()`. When we have additional
type parameters, the number of monomorphized implementations the compiler
generates does not grow drastically, since the compiler will only generate an
implementation if the function is called with unparametrized substitutions
(i.e., substitutions where none of the substituted types are themselves
parametrized).

However, with trait objects we have to make a table containing _every_ object
that implements the trait. Now, if it has type parameters, we need to add
implementations for every type that implements the trait, and there could
theoretically be an infinite number of types.

For example, with:

```
trait Trait {
    fn foo<T>(&self, on: T);
    // more methods
}
impl Trait for String {
    fn foo<T>(&self, on: T) {
        // implementation 1
    }
}
impl Trait for u8 {
    fn foo<T>(&self, on: T) {
        // implementation 2
    }
}
// 8 more implementations
```

Now, if we have the following code:

```
fn call_foo(thing: Box<Trait>) {
    thing.foo(true); // this could be any one of the 8 types above
    thing.foo(1);
    thing.foo("hello");
}
```

we don't just need to create a table of all implementations of all methods of
`Trait`, we need to create such a table, for each different type fed to
`foo()`. In this case this turns out to be (10 types implementing `Trait`)*(3
types being fed to `foo()`) = 30 implementations!

With real world traits these numbers can grow drastically.

To fix this, it is suggested to use a `where Self: Sized` bound similar to the
fix for the sub-error above if you do not intend to call the method with type
parameters:

```
trait Trait {
    fn foo<T>(&self, on: T) where Self: Sized;
    // more methods
}
```

If this is not an option, consider replacing the type parameter with another
trait object (e.g. if `T: OtherTrait`, use `on: Box<OtherTrait>`). If the number
of types you intend to feed to this method is limited, consider manually listing
out the methods of different types.

### Method has no receiver

Methods that do not take a `self` parameter can't be called since there won't be
a way to get a pointer to the method table for them

```
trait Foo {
    fn foo() -> u8;
}
```

This could be called as `<Foo as Foo>::foo()`, which would not be able to pick
an implementation.

Adding a `Self: Sized` bound to these methods will generally make this compile.

```
trait Foo {
    fn foo() -> u8 where Self: Sized;
}
```

### The trait cannot use `Self` as a type parameter in the supertrait listing

This is similar to the second sub-error, but subtler. It happens in situations
like the following:

```
trait Super<A> {}

trait Trait: Super<Self> {
}

struct Foo;

impl Super<Foo> for Foo{}

impl Trait for Foo {}
```

Here, the supertrait might have methods as follows:

```
trait Super<A> {
    fn get_a(&self) -> A; // note that this is object safe!
}
```

If the trait `Foo` was deriving from something like `Super<String>` or
`Super<T>` (where `Foo` itself is `Foo<T>`), this is okay, because given a type
`get_a()` will definitely return an object of that type.

However, if it derives from `Super<Self>`, even though `Super` is object safe,
the method `get_a()` would return an object of unknown type when called on the
function. `Self` type parameters let us make object safe traits no longer safe,
so they are forbidden when specifying supertraits.

There's no easy fix for this, generally code will need to be refactored so that
you no longer need to derive from `Super<Self>`.
"####,

E0109: r##"
You tried to give a type parameter to a type which doesn't need it. Erroneous
code example:

```
type X = u32<i32>; // error: type parameters are not allowed on this type
```

Please check that you used the correct type and recheck its definition. Perhaps
it doesn't need the type parameter.

Example:

```
type X = u32; // this compiles
```

Note that type parameters for enum-variant constructors go after the variant,
not after the enum (Option::None::<u32>, not Option::<u32>::None).
"##,

E0110: r##"
You tried to give a lifetime parameter to a type which doesn't need it.
Erroneous code example:

```
type X = u32<'static>; // error: lifetime parameters are not allowed on
                       //        this type
```

Please check that the correct type was used and recheck its definition; perhaps
it doesn't need the lifetime parameter. Example:

```
type X = u32; // ok!
```
"##,

E0133: r##"
Using unsafe functionality, is potentially dangerous and disallowed
by safety checks. Examples:

- Dereferencing raw pointers
- Calling functions via FFI
- Calling functions marked unsafe

These safety checks can be relaxed for a section of the code
by wrapping the unsafe instructions with an `unsafe` block. For instance:

```
unsafe fn f() { return; }

fn main() {
    unsafe { f(); }
}
```

See also https://doc.rust-lang.org/book/unsafe.html
"##,

// This shouldn't really ever trigger since the repeated value error comes first
E0136: r##"
A binary can only have one entry point, and by default that entry point is the
function `main()`. If there are multiple such functions, please rename one.
"##,

E0137: r##"
This error indicates that the compiler found multiple functions with the
`#[main]` attribute. This is an error because there must be a unique entry
point into a Rust program.
"##,

E0138: r##"
This error indicates that the compiler found multiple functions with the
`#[start]` attribute. This is an error because there must be a unique entry
point into a Rust program.
"##,

// FIXME link this to the relevant turpl chapters for instilling fear of the
//       transmute gods in the user
E0139: r##"
There are various restrictions on transmuting between types in Rust; for example
types being transmuted must have the same size. To apply all these restrictions,
the compiler must know the exact types that may be transmuted. When type
parameters are involved, this cannot always be done.

So, for example, the following is not allowed:

```
struct Foo<T>(Vec<T>)

fn foo<T>(x: Vec<T>) {
    // we are transmuting between Vec<T> and Foo<T> here
    let y: Foo<T> = unsafe { transmute(x) };
    // do something with y
}
```

In this specific case there's a good chance that the transmute is harmless (but
this is not guaranteed by Rust). However, when alignment and enum optimizations
come into the picture, it's quite likely that the sizes may or may not match
with different type parameter substitutions. It's not possible to check this for
_all_ possible types, so `transmute()` simply only accepts types without any
unsubstituted type parameters.

If you need this, there's a good chance you're doing something wrong. Keep in
mind that Rust doesn't guarantee much about the layout of different structs
(even two structs with identical declarations may have different layouts). If
there is a solution that avoids the transmute entirely, try it instead.

If it's possible, hand-monomorphize the code by writing the function for each
possible type substitution. It's possible to use traits to do this cleanly,
for example:

```
trait MyTransmutableType {
    fn transmute(Vec<Self>) -> Foo<Self>
}

impl MyTransmutableType for u8 {
    fn transmute(x: Foo<u8>) -> Vec<u8> {
        transmute(x)
    }
}
impl MyTransmutableType for String {
    fn transmute(x: Foo<String>) -> Vec<String> {
        transmute(x)
    }
}
// ... more impls for the types you intend to transmute

fn foo<T: MyTransmutableType>(x: Vec<T>) {
    let y: Foo<T> = <T as MyTransmutableType>::transmute(x);
    // do something with y
}
```

Each impl will be checked for a size match in the transmute as usual, and since
there are no unbound type parameters involved, this should compile unless there
is a size mismatch in one of the impls.

It is also possible to manually transmute:

```
ptr::read(&v as *const _ as *const SomeType) // `v` transmuted to `SomeType`
```

Note that this does not move `v` (unlike `transmute`), and may need a
call to `mem::forget(v)` in case you want to avoid destructors being called.
"##,

E0152: r##"
Lang items are already implemented in the standard library. Unless you are
writing a free-standing application (e.g. a kernel), you do not need to provide
them yourself.

You can build a free-standing crate by adding `#![no_std]` to the crate
attributes:

```
#![feature(no_std)]
#![no_std]
```

See also https://doc.rust-lang.org/book/no-stdlib.html
"##,

E0158: r##"
`const` and `static` mean different things. A `const` is a compile-time
constant, an alias for a literal value. This property means you can match it
directly within a pattern.

The `static` keyword, on the other hand, guarantees a fixed location in memory.
This does not always mean that the value is constant. For example, a global
mutex can be declared `static` as well.

If you want to match against a `static`, consider using a guard instead:

```
static FORTY_TWO: i32 = 42;
match Some(42) {
    Some(x) if x == FORTY_TWO => ...
    ...
}
```
"##,

E0161: r##"
In Rust, you can only move a value when its size is known at compile time.

To work around this restriction, consider "hiding" the value behind a reference:
either `&x` or `&mut x`. Since a reference has a fixed size, this lets you move
it around as usual.
"##,

E0162: r##"
An if-let pattern attempts to match the pattern, and enters the body if the
match was successful. If the match is irrefutable (when it cannot fail to
match), use a regular `let`-binding instead. For instance:

```
struct Irrefutable(i32);
let irr = Irrefutable(0);

// This fails to compile because the match is irrefutable.
if let Irrefutable(x) = irr {
    // This body will always be executed.
    foo(x);
}

// Try this instead:
let Irrefutable(x) = irr;
foo(x);
```
"##,

E0165: r##"
A while-let pattern attempts to match the pattern, and enters the body if the
match was successful. If the match is irrefutable (when it cannot fail to
match), use a regular `let`-binding inside a `loop` instead. For instance:

```
struct Irrefutable(i32);
let irr = Irrefutable(0);

// This fails to compile because the match is irrefutable.
while let Irrefutable(x) = irr {
    ...
}

// Try this instead:
loop {
    let Irrefutable(x) = irr;
    ...
}
```
"##,

E0170: r##"
Enum variants are qualified by default. For example, given this type:

```
enum Method {
    GET,
    POST
}
```

you would match it using:

```
match m {
    Method::GET => ...
    Method::POST => ...
}
```

If you don't qualify the names, the code will bind new variables named "GET" and
"POST" instead. This behavior is likely not what you want, so `rustc` warns when
that happens.

Qualified names are good practice, and most code works well with them. But if
you prefer them unqualified, you can import the variants into scope:

```
use Method::*;
enum Method { GET, POST }
```

If you want others to be able to import variants from your module directly, use
`pub use`:

```
pub use Method::*;
enum Method { GET, POST }
```
"##,

E0261: r##"
When using a lifetime like `'a` in a type, it must be declared before being
used.

These two examples illustrate the problem:

```
// error, use of undeclared lifetime name `'a`
fn foo(x: &'a str) { }

struct Foo {
    // error, use of undeclared lifetime name `'a`
    x: &'a str,
}
```

These can be fixed by declaring lifetime parameters:

```
fn foo<'a>(x: &'a str) { }

struct Foo<'a> {
    x: &'a str,
}
```
"##,

E0262: r##"
Declaring certain lifetime names in parameters is disallowed. For example,
because the `'static` lifetime is a special built-in lifetime name denoting
the lifetime of the entire program, this is an error:

```
// error, invalid lifetime parameter name `'static`
fn foo<'static>(x: &'static str) { }
```
"##,

E0263: r##"
A lifetime name cannot be declared more than once in the same scope. For
example:

```
// error, lifetime name `'a` declared twice in the same scope
fn foo<'a, 'b, 'a>(x: &'a str, y: &'b str) { }
```
"##,

E0265: r##"
This error indicates that a static or constant references itself.
All statics and constants need to resolve to a value in an acyclic manner.

For example, neither of the following can be sensibly compiled:

```
const X: u32 = X;
```

```
const X: u32 = Y;
const Y: u32 = X;
```
"##,

E0267: r##"
This error indicates the use of a loop keyword (`break` or `continue`) inside a
closure but outside of any loop. Erroneous code example:

```
let w = || { break; }; // error: `break` inside of a closure
```

`break` and `continue` keywords can be used as normal inside closures as long as
they are also contained within a loop. To halt the execution of a closure you
should instead use a return statement. Example:

```
let w = || {
    for _ in 0..10 {
        break;
    }
};

w();
```
"##,

E0268: r##"
This error indicates the use of a loop keyword (`break` or `continue`) outside
of a loop. Without a loop to break out of or continue in, no sensible action can
be taken. Erroneous code example:

```
fn some_func() {
    break; // error: `break` outside of loop
}
```

Please verify that you are using `break` and `continue` only in loops. Example:

```
fn some_func() {
    for _ in 0..10 {
        break; // ok!
    }
}
```
"##,

E0269: r##"
Functions must eventually return a value of their return type. For example, in
the following function

```
fn foo(x: u8) -> u8 {
    if x > 0 {
        x // alternatively, `return x`
    }
    // nothing here
}
```

if the condition is true, the value `x` is returned, but if the condition is
false, control exits the `if` block and reaches a place where nothing is being
returned. All possible control paths must eventually return a `u8`, which is not
happening here.

An easy fix for this in a complicated function is to specify a default return
value, if possible:

```
fn foo(x: u8) -> u8 {
    if x > 0 {
        x // alternatively, `return x`
    }
    // lots of other if branches
    0 // return 0 if all else fails
}
```

It is advisable to find out what the unhandled cases are and check for them,
returning an appropriate value or panicking if necessary.
"##,

E0270: r##"
Rust lets you define functions which are known to never return, i.e. are
'diverging', by marking its return type as `!`.

For example, the following functions never return:

```
fn foo() -> ! {
    loop {}
}

fn bar() -> ! {
    foo() // foo() is diverging, so this will diverge too
}

fn baz() -> ! {
    panic!(); // this macro internally expands to a call to a diverging function
}

```

Such functions can be used in a place where a value is expected without
returning a value of that type,  for instance:

```
let y = match x {
    1 => 1,
    2 => 4,
    _ => foo() // diverging function called here
};
println!("{}", y)
```

If the third arm of the match block is reached, since `foo()` doesn't ever
return control to the match block, it is fine to use it in a place where an
integer was expected. The `match` block will never finish executing, and any
point where `y` (like the print statement) is needed will not be reached.

However, if we had a diverging function that actually does finish execution

```
fn foo() -> {
    loop {break;}
}
```

then we would have an unknown value for `y` in the following code:

```
let y = match x {
    1 => 1,
    2 => 4,
    _ => foo()
};
println!("{}", y);
```

In the previous example, the print statement was never reached when the wildcard
match arm was hit, so we were okay with `foo()` not returning an integer that we
could set to `y`. But in this example, `foo()` actually does return control, so
the print statement will be executed with an uninitialized value.

Obviously we cannot have functions which are allowed to be used in such
positions and yet can return control. So, if you are defining a function that
returns `!`, make sure that there is no way for it to actually finish executing.
"##,

E0271: r##"
This is because of a type mismatch between the associated type of some
trait (e.g. `T::Bar`, where `T` implements `trait Quux { type Bar; }`)
and another type `U` that is required to be equal to `T::Bar`, but is not.
Examples follow.

Here is a basic example:

```
trait Trait { type AssociatedType; }
fn foo<T>(t: T) where T: Trait<AssociatedType=u32> {
    println!("in foo");
}
impl Trait for i8 { type AssociatedType = &'static str; }
foo(3_i8);
```

Here is that same example again, with some explanatory comments:

```
trait Trait { type AssociatedType; }

fn foo<T>(t: T) where T: Trait<AssociatedType=u32> {
//                    ~~~~~~~~ ~~~~~~~~~~~~~~~~~~
//                        |            |
//         This says `foo` can         |
//           only be used with         |
//              some type that         |
//         implements `Trait`.         |
//                                     |
//                             This says not only must
//                             `T` be an impl of `Trait`
//                             but also that the impl
//                             must assign the type `u32`
//                             to the associated type.
    println!("in foo");
}

impl Trait for i8 { type AssociatedType = &'static str; }
~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//      |                             |
// `i8` does have                     |
// implementation                     |
// of `Trait`...                      |
//                     ... but it is an implementation
//                     that assigns `&'static str` to
//                     the associated type.

foo(3_i8);
// Here, we invoke `foo` with an `i8`, which does not satisfy
// the constraint `<i8 as Trait>::AssociatedType=u32`, and
// therefore the type-checker complains with this error code.
```

Here is a more subtle instance of the same problem, that can
arise with for-loops in Rust:

```
let vs: Vec<i32> = vec![1, 2, 3, 4];
for v in &vs {
    match v {
        1 => {}
        _ => {}
    }
}
```

The above fails because of an analogous type mismatch,
though may be harder to see. Again, here are some
explanatory comments for the same example:

```
{
    let vs = vec![1, 2, 3, 4];

    // `for`-loops use a protocol based on the `Iterator`
    // trait. Each item yielded in a `for` loop has the
    // type `Iterator::Item` -- that is,I `Item` is the
    // associated type of the concrete iterator impl.
    for v in &vs {
//      ~    ~~~
//      |     |
//      |    We borrow `vs`, iterating over a sequence of
//      |    *references* of type `&Elem` (where `Elem` is
//      |    vector's element type). Thus, the associated
//      |    type `Item` must be a reference `&`-type ...
//      |
//  ... and `v` has the type `Iterator::Item`, as dictated by
//  the `for`-loop protocol ...

        match v {
            1 => {}
//          ~
//          |
// ... but *here*, `v` is forced to have some integral type;
// only types like `u8`,`i8`,`u16`,`i16`, et cetera can
// match the pattern `1` ...

            _ => {}
        }

// ... therefore, the compiler complains, because it sees
// an attempt to solve the equations
// `some integral-type` = type-of-`v`
//                      = `Iterator::Item`
//                      = `&Elem` (i.e. `some reference type`)
//
// which cannot possibly all be true.

    }
}
```

To avoid those issues, you have to make the types match correctly.
So we can fix the previous examples like this:

```
// Basic Example:
trait Trait { type AssociatedType; }
fn foo<T>(t: T) where T: Trait<AssociatedType = &'static str> {
    println!("in foo");
}
impl Trait for i8 { type AssociatedType = &'static str; }
foo(3_i8);

// For-Loop Example:
let vs = vec![1, 2, 3, 4];
for v in &vs {
    match v {
        &1 => {}
        _ => {}
    }
}
```
"##,

E0272: r##"
The `#[rustc_on_unimplemented]` attribute lets you specify a custom error
message for when a particular trait isn't implemented on a type placed in a
position that needs that trait. For example, when the following code is
compiled:

```
fn foo<T: Index<u8>>(x: T){}

#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
trait Index<Idx> { ... }

foo(true); // `bool` does not implement `Index<u8>`
```

there will be an error about `bool` not implementing `Index<u8>`, followed by a
note saying "the type `bool` cannot be indexed by `u8`".

As you can see, you can specify type parameters in curly braces for substitution
with the actual types (using the regular format string syntax) in a given
situation. Furthermore, `{Self}` will substitute to the type (in this case,
`bool`) that we tried to use.

This error appears when the curly braces contain an identifier which doesn't
match with any of the type parameters or the string `Self`. This might happen if
you misspelled a type parameter, or if you intended to use literal curly braces.
If it is the latter, escape the curly braces with a second curly brace of the
same type; e.g. a literal `{` is `{{`
"##,

E0273: r##"
The `#[rustc_on_unimplemented]` attribute lets you specify a custom error
message for when a particular trait isn't implemented on a type placed in a
position that needs that trait. For example, when the following code is
compiled:

```
fn foo<T: Index<u8>>(x: T){}

#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
trait Index<Idx> { ... }

foo(true); // `bool` does not implement `Index<u8>`
```

there will be an error about `bool` not implementing `Index<u8>`, followed by a
note saying "the type `bool` cannot be indexed by `u8`".

As you can see, you can specify type parameters in curly braces for substitution
with the actual types (using the regular format string syntax) in a given
situation. Furthermore, `{Self}` will substitute to the type (in this case,
`bool`) that we tried to use.

This error appears when the curly braces do not contain an identifier. Please
add one of the same name as a type parameter. If you intended to use literal
braces, use `{{` and `}}` to escape them.
"##,

E0274: r##"
The `#[rustc_on_unimplemented]` attribute lets you specify a custom error
message for when a particular trait isn't implemented on a type placed in a
position that needs that trait. For example, when the following code is
compiled:

```
fn foo<T: Index<u8>>(x: T){}

#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
trait Index<Idx> { ... }

foo(true); // `bool` does not implement `Index<u8>`
```

there will be an error about `bool` not implementing `Index<u8>`, followed by a
note saying "the type `bool` cannot be indexed by `u8`".

For this to work, some note must be specified. An empty attribute will not do
anything, please remove the attribute or add some helpful note for users of the
trait.
"##,

E0275: r##"
This error occurs when there was a recursive trait requirement that overflowed
before it could be evaluated. Often this means that there is unbounded recursion
in resolving some type bounds.

For example, in the following code

```
trait Foo {}

struct Bar<T>(T);

impl<T> Foo for T where Bar<T>: Foo {}
```

to determine if a `T` is `Foo`, we need to check if `Bar<T>` is `Foo`. However,
to do this check, we need to determine that `Bar<Bar<T>>` is `Foo`. To determine
this, we check if `Bar<Bar<Bar<T>>>` is `Foo`, and so on. This is clearly a
recursive requirement that can't be resolved directly.

Consider changing your trait bounds so that they're less self-referential.
"##,

E0276: r##"
This error occurs when a bound in an implementation of a trait does not match
the bounds specified in the original trait. For example:

```
trait Foo {
 fn foo<T>(x: T);
}

impl Foo for bool {
 fn foo<T>(x: T) where T: Copy {}
}
```

Here, all types implementing `Foo` must have a method `foo<T>(x: T)` which can
take any type `T`. However, in the `impl` for `bool`, we have added an extra
bound that `T` is `Copy`, which isn't compatible with the original trait.

Consider removing the bound from the method or adding the bound to the original
method definition in the trait.
"##,

E0277: r##"
You tried to use a type which doesn't implement some trait in a place which
expected that trait. Erroneous code example:

```
// here we declare the Foo trait with a bar method
trait Foo {
    fn bar(&self);
}

// we now declare a function which takes an object implementing the Foo trait
fn some_func<T: Foo>(foo: T) {
    foo.bar();
}

fn main() {
    // we now call the method with the i32 type, which doesn't implement
    // the Foo trait
    some_func(5i32); // error: the trait `Foo` is not implemented for the
                     //     type `i32`
}
```

In order to fix this error, verify that the type you're using does implement
the trait. Example:

```
trait Foo {
    fn bar(&self);
}

fn some_func<T: Foo>(foo: T) {
    foo.bar(); // we can now use this method since i32 implements the
               // Foo trait
}

// we implement the trait on the i32 type
impl Foo for i32 {
    fn bar(&self) {}
}

fn main() {
    some_func(5i32); // ok!
}
```
"##,

E0281: r##"
You tried to supply a type which doesn't implement some trait in a location
which expected that trait. This error typically occurs when working with
`Fn`-based types. Erroneous code example:

```
fn foo<F: Fn()>(x: F) { }

fn main() {
    // type mismatch: the type ... implements the trait `core::ops::Fn<(_,)>`,
    // but the trait `core::ops::Fn<()>` is required (expected (), found tuple
    // [E0281]
    foo(|y| { });
}
```

The issue in this case is that `foo` is defined as accepting a `Fn` with no
arguments, but the closure we attempted to pass to it requires one argument.
"##,

E0282: r##"
This error indicates that type inference did not result in one unique possible
type, and extra information is required. In most cases this can be provided
by adding a type annotation. Sometimes you need to specify a generic type
parameter manually.

A common example is the `collect` method on `Iterator`. It has a generic type
parameter with a `FromIterator` bound, which for a `char` iterator is
implemented by `Vec` and `String` among others. Consider the following snippet
that reverses the characters of a string:

```
let x = "hello".chars().rev().collect();
```

In this case, the compiler cannot infer what the type of `x` should be:
`Vec<char>` and `String` are both suitable candidates. To specify which type to
use, you can use a type annotation on `x`:

```
let x: Vec<char> = "hello".chars().rev().collect();
```

It is not necessary to annotate the full type. Once the ambiguity is resolved,
the compiler can infer the rest:

```
let x: Vec<_> = "hello".chars().rev().collect();
```

Another way to provide the compiler with enough information, is to specify the
generic type parameter:

```
let x = "hello".chars().rev().collect::<Vec<char>>();
```

Again, you need not specify the full type if the compiler can infer it:

```
let x = "hello".chars().rev().collect::<Vec<_>>();
```

Apart from a method or function with a generic type parameter, this error can
occur when a type parameter of a struct or trait cannot be inferred. In that
case it is not always possible to use a type annotation, because all candidates
have the same return type. For instance:

```
struct Foo<T> {
    // Some fields omitted.
}

impl<T> Foo<T> {
    fn bar() -> i32 {
        0
    }

    fn baz() {
        let number = Foo::bar();
    }
}
```

This will fail because the compiler does not know which instance of `Foo` to
call `bar` on. Change `Foo::bar()` to `Foo::<T>::bar()` to resolve the error.
"##,

E0296: r##"
This error indicates that the given recursion limit could not be parsed. Ensure
that the value provided is a positive integer between quotes, like so:

```
#![recursion_limit="1000"]
```
"##,

E0297: r##"
Patterns used to bind names must be irrefutable. That is, they must guarantee
that a name will be extracted in all cases. Instead of pattern matching the
loop variable, consider using a `match` or `if let` inside the loop body. For
instance:

```
// This fails because `None` is not covered.
for Some(x) in xs {
    ...
}

// Match inside the loop instead:
for item in xs {
    match item {
        Some(x) => ...
        None => ...
    }
}

// Or use `if let`:
for item in xs {
    if let Some(x) = item {
        ...
    }
}
```
"##,

E0301: r##"
Mutable borrows are not allowed in pattern guards, because matching cannot have
side effects. Side effects could alter the matched object or the environment
on which the match depends in such a way, that the match would not be
exhaustive. For instance, the following would not match any arm if mutable
borrows were allowed:

```
match Some(()) {
    None => { },
    option if option.take().is_none() => { /* impossible, option is `Some` */ },
    Some(_) => { } // When the previous match failed, the option became `None`.
}
```
"##,

E0302: r##"
Assignments are not allowed in pattern guards, because matching cannot have
side effects. Side effects could alter the matched object or the environment
on which the match depends in such a way, that the match would not be
exhaustive. For instance, the following would not match any arm if assignments
were allowed:

```
match Some(()) {
    None => { },
    option if { option = None; false } { },
    Some(_) => { } // When the previous match failed, the option became `None`.
}
```
"##,

E0303: r##"
In certain cases it is possible for sub-bindings to violate memory safety.
Updates to the borrow checker in a future version of Rust may remove this
restriction, but for now patterns must be rewritten without sub-bindings.

```
// Before.
match Some("hi".to_string()) {
    ref op_string_ref @ Some(ref s) => ...
    None => ...
}

// After.
match Some("hi".to_string()) {
    Some(ref s) => {
        let op_string_ref = &Some(s);
        ...
    }
    None => ...
}
```

The `op_string_ref` binding has type `&Option<&String>` in both cases.

See also https://github.com/rust-lang/rust/issues/14587
"##,

E0306: r##"
In an array literal `[x; N]`, `N` is the number of elements in the array. This
number cannot be negative.
"##,

E0307: r##"
The length of an array is part of its type. For this reason, this length must be
a compile-time constant.
"##,

E0308: r##"
This error occurs when the compiler was unable to infer the concrete type of a
variable. It can occur for several cases, the most common of which is a
mismatch in the expected type that the compiler inferred for a variable's
initializing expression, and the actual type explicitly assigned to the
variable.

For example:

```
let x: i32 = "I am not a number!";
//     ~~~   ~~~~~~~~~~~~~~~~~~~~
//      |             |
//      |    initializing expression;
//      |    compiler infers type `&str`
//      |
//    type `i32` assigned to variable `x`
```
"##,

E0309: r##"
Types in type definitions have lifetimes associated with them that represent
how long the data stored within them is guaranteed to be live. This lifetime
must be as long as the data needs to be alive, and missing the constraint that
denotes this will cause this error.

```
// This won't compile because T is not constrained, meaning the data
// stored in it is not guaranteed to last as long as the reference
struct Foo<'a, T> {
    foo: &'a T
}

// This will compile, because it has the constraint on the type parameter
struct Foo<'a, T: 'a> {
    foo: &'a T
}
```
"##,

E0310: r##"
Types in type definitions have lifetimes associated with them that represent
how long the data stored within them is guaranteed to be live. This lifetime
must be as long as the data needs to be alive, and missing the constraint that
denotes this will cause this error.

```
// This won't compile because T is not constrained to the static lifetime
// the reference needs
struct Foo<T> {
    foo: &'static T
}

// This will compile, because it has the constraint on the type parameter
struct Foo<T: 'static> {
    foo: &'static T
}
```
"##,

E0378: r##"
Method calls that aren't calls to inherent `const` methods are disallowed
in statics, constants, and constant functions.

For example:

```
const BAZ: i32 = Foo(25).bar(); // error, `bar` isn't `const`

struct Foo(i32);

impl Foo {
    const fn foo(&self) -> i32 {
        self.bar() // error, `bar` isn't `const`
    }

    fn bar(&self) -> i32 { self.0 }
}
```

For more information about `const fn`'s, see [RFC 911].

[RFC 911]: https://github.com/rust-lang/rfcs/blob/master/text/0911-const-fn.md
"##,

E0394: r##"
From [RFC 246]:

 > It is invalid for a static to reference another static by value. It is
 > required that all references be borrowed.

[RFC 246]: https://github.com/rust-lang/rfcs/pull/246
"##,

E0395: r##"
The value assigned to a constant expression must be known at compile time,
which is not the case when comparing raw pointers. Erroneous code example:

```
static foo: i32 = 42;
static bar: i32 = 43;

static baz: bool = { (&foo as *const i32) == (&bar as *const i32) };
// error: raw pointers cannot be compared in statics!
```

Please check that the result of the comparison can be determined at compile time
or isn't assigned to a constant expression. Example:

```
static foo: i32 = 42;
static bar: i32 = 43;

let baz: bool = { (&foo as *const i32) == (&bar as *const i32) };
// baz isn't a constant expression so it's ok
```
"##,

E0396: r##"
The value assigned to a constant expression must be known at compile time,
which is not the case when dereferencing raw pointers. Erroneous code
example:

```
const foo: i32 = 42;
const baz: *const i32 = (&foo as *const i32);

const deref: i32 = *baz;
// error: raw pointers cannot be dereferenced in constants
```

To fix this error, please do not assign this value to a constant expression.
Example:

```
const foo: i32 = 42;
const baz: *const i32 = (&foo as *const i32);

unsafe { let deref: i32 = *baz; }
// baz isn't a constant expression so it's ok
```

You'll also note that this assignment must be done in an unsafe block!
"##,

E0397: r##"
It is not allowed for a mutable static to allocate or have destructors. For
example:

```
// error: mutable statics are not allowed to have boxes
static mut FOO: Option<Box<usize>> = None;

// error: mutable statics are not allowed to have destructors
static mut BAR: Option<Vec<i32>> = None;
```
"##,

E0398: r##"
In Rust 1.3, the default object lifetime bounds are expected to
change, as described in RFC #1156 [1]. You are getting a warning
because the compiler thinks it is possible that this change will cause
a compilation error in your code. It is possible, though unlikely,
that this is a false alarm.

The heart of the change is that where `&'a Box<SomeTrait>` used to
default to `&'a Box<SomeTrait+'a>`, it now defaults to `&'a
Box<SomeTrait+'static>` (here, `SomeTrait` is the name of some trait
type). Note that the only types which are affected are references to
boxes, like `&Box<SomeTrait>` or `&[Box<SomeTrait>]`.  More common
types like `&SomeTrait` or `Box<SomeTrait>` are unaffected.

To silence this warning, edit your code to use an explicit bound.
Most of the time, this means that you will want to change the
signature of a function that you are calling. For example, if
the error is reported on a call like `foo(x)`, and `foo` is
defined as follows:

```
fn foo(arg: &Box<SomeTrait>) { ... }
```

you might change it to:

```
fn foo<'a>(arg: &Box<SomeTrait+'a>) { ... }
```

This explicitly states that you expect the trait object `SomeTrait` to
contain references (with a maximum lifetime of `'a`).

[1]: https://github.com/rust-lang/rfcs/pull/1156
"##,

E0400: r##"
A user-defined dereference was attempted in an invalid context. Erroneous
code example:

```
use std::ops::Deref;

struct A;

impl Deref for A {
    type Target = str;

    fn deref(&self)-> &str { "foo" }
}

const S: &'static str = &A;
// error: user-defined dereference operators are not allowed in constants

fn main() {
    let foo = S;
}
```

You cannot directly use a dereference operation whilst initializing a constant
or a static. To fix this error, restructure your code to avoid this dereference,
perhaps moving it inline:

```
use std::ops::Deref;

struct A;

impl Deref for A {
    type Target = str;

    fn deref(&self)-> &str { "foo" }
}

fn main() {
    let foo : &str = &A;
}
```
"##,

E0452: r##"
An invalid lint attribute has been given. Erroneous code example:

```
#![allow(foo = "")] // error: malformed lint attribute
```

Lint attributes only accept a list of identifiers (where each identifier is a
lint name). Ensure the attribute is of this form:

```
#![allow(foo)] // ok!
// or:
#![allow(foo, foo2)] // ok!
```
"##,

E0492: r##"
A borrow of a constant containing interior mutability was attempted. Erroneous
code example:

```
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};

const A: AtomicUsize = ATOMIC_USIZE_INIT;
static B: &'static AtomicUsize = &A;
// error: cannot borrow a constant which contains interior mutability, create a
//        static instead
```

A `const` represents a constant value that should never change. If one takes
a `&` reference to the constant, then one is taking a pointer to some memory
location containing the value. Normally this is perfectly fine: most values
can't be changed via a shared `&` pointer, but interior mutability would allow
it. That is, a constant value could be mutated. On the other hand, a `static` is
explicitly a single memory location, which can be mutated at will.

So, in order to solve this error, either use statics which are `Sync`:

```
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};

static A: AtomicUsize = ATOMIC_USIZE_INIT;
static B: &'static AtomicUsize = &A; // ok!
```

You can also have this error while using a cell type:

```
#![feature(const_fn)]

use std::cell::Cell;

const A: Cell<usize> = Cell::new(1);
const B: &'static Cell<usize> = &A;
// error: cannot borrow a constant which contains interior mutability, create
//        a static instead

// or:
struct C { a: Cell<usize> }

const D: C = C { a: Cell::new(1) };
const E: &'static Cell<usize> = &D.a; // error

// or:
const F: &'static C = &D; // error
```

This is because cell types do operations that are not thread-safe. Due to this,
they don't implement Sync and thus can't be placed in statics. In this
case, `StaticMutex` would work just fine, but it isn't stable yet:
https://doc.rust-lang.org/nightly/std/sync/struct.StaticMutex.html

However, if you still wish to use these types, you can achieve this by an unsafe
wrapper:

```
#![feature(const_fn)]

use std::cell::Cell;
use std::marker::Sync;

struct NotThreadSafe<T> {
    value: Cell<T>,
}

unsafe impl<T> Sync for NotThreadSafe<T> {}

static A: NotThreadSafe<usize> = NotThreadSafe { value : Cell::new(1) };
static B: &'static NotThreadSafe<usize> = &A; // ok!
```

Remember this solution is unsafe! You will have to ensure that accesses to the
cell are synchronized.
"##,

E0493: r##"
A type with a destructor was assigned to an invalid type of variable. Erroneous
code example:

```
struct Foo {
    a: u32
}

impl Drop for Foo {
    fn drop(&mut self) {}
}

const F : Foo = Foo { a : 0 };
// error: constants are not allowed to have destructors
static S : Foo = Foo { a : 0 };
// error: statics are not allowed to have destructors
```

To solve this issue, please use a type which does allow the usage of type with
destructors.
"##,

E0494: r##"
A reference of an interior static was assigned to another const/static.
Erroneous code example:

```
struct Foo {
    a: u32
}

static S : Foo = Foo { a : 0 };
static A : &'static u32 = &S.a;
// error: cannot refer to the interior of another static, use a
//        constant instead
```

The "base" variable has to be a const if you want another static/const variable
to refer to one of its fields. Example:

```
struct Foo {
    a: u32
}

const S : Foo = Foo { a : 0 };
static A : &'static u32 = &S.a; // ok!
```
"##,

E0496: r##"
A lifetime name is shadowing another lifetime name. Erroneous code example:

```
struct Foo<'a> {
    a: &'a i32,
}

impl<'a> Foo<'a> {
    fn f<'a>(x: &'a i32) { // error: lifetime name `'a` shadows a lifetime
                           //        name that is already in scope
    }
}
```

Please change the name of one of the lifetimes to remove this error. Example:

```
struct Foo<'a> {
    a: &'a i32,
}

impl<'a> Foo<'a> {
    fn f<'b>(x: &'b i32) { // ok!
    }
}

fn main() {
}
```
"##,

E0497: r##"
A stability attribute was used outside of the standard library. Erroneous code
example:

```
#[stable] // error: stability attributes may not be used outside of the
          //        standard library
fn foo() {}
```

It is not possible to use stability attributes outside of the standard library.
Also, for now, it is not possible to write deprecation messages either.
"##,

E0517: r##"
This error indicates that a `#[repr(..)]` attribute was placed on an unsupported
item.

Examples of erroneous code:

```
#[repr(C)]
type Foo = u8;

#[repr(packed)]
enum Foo {Bar, Baz}

#[repr(u8)]
struct Foo {bar: bool, baz: bool}

#[repr(C)]
impl Foo {
    ...
}
```

 - The `#[repr(C)]` attribute can only be placed on structs and enums
 - The `#[repr(packed)]` and `#[repr(simd)]` attributes only work on structs
 - The `#[repr(u8)]`, `#[repr(i16)]`, etc attributes only work on enums

These attributes do not work on typedefs, since typedefs are just aliases.

Representations like `#[repr(u8)]`, `#[repr(i64)]` are for selecting the
discriminant size for C-like enums (when there is no associated data, e.g. `enum
Color {Red, Blue, Green}`), effectively setting the size of the enum to the size
of the provided type. Such an enum can be cast to a value of the same type as
well. In short, `#[repr(u8)]` makes the enum behave like an integer with a
constrained set of allowed values.

Only C-like enums can be cast to numerical primitives, so this attribute will
not apply to structs.

`#[repr(packed)]` reduces padding to make the struct size smaller. The
representation of enums isn't strictly defined in Rust, and this attribute won't
work on enums.

`#[repr(simd)]` will give a struct consisting of a homogenous series of machine
types (i.e. `u8`, `i32`, etc) a representation that permits vectorization via
SIMD. This doesn't make much sense for enums since they don't consist of a
single list of data.
"##,

E0518: r##"
This error indicates that an `#[inline(..)]` attribute was incorrectly placed on
something other than a function or method.

Examples of erroneous code:

```
#[inline(always)]
struct Foo;

#[inline(never)]
impl Foo {
    ...
}
```

`#[inline]` hints the compiler whether or not to attempt to inline a method or
function. By default, the compiler does a pretty good job of figuring this out
itself, but if you feel the need for annotations, `#[inline(always)]` and
`#[inline(never)]` can override or force the compiler's decision.

If you wish to apply this attribute to all methods in an impl, manually annotate
each method; it is not possible to annotate the entire impl with an `#[inline]`
attribute.
"##,

}


register_diagnostics! {
    // E0006 // merged with E0005
//  E0134,
//  E0135,
    E0229, // associated type bindings are not allowed here
    E0264, // unknown external lang item
    E0278, // requirement is not satisfied
    E0279, // requirement is not satisfied
    E0280, // requirement is not satisfied
    E0283, // cannot resolve type
    E0284, // cannot resolve type
    E0285, // overflow evaluation builtin bounds
    E0298, // mismatched types between arms
    E0299, // mismatched types between arms
    E0300, // unexpanded macro
    E0304, // expected signed integer constant
    E0305, // expected constant
    E0311, // thing may not live long enough
    E0312, // lifetime of reference outlives lifetime of borrowed content
    E0313, // lifetime of borrowed pointer outlives lifetime of captured variable
    E0314, // closure outlives stack frame
    E0315, // cannot invoke closure outside of its lifetime
    E0316, // nested quantification of lifetimes
    E0453, // overruled by outer forbid
    E0471, // constant evaluation error: ..
    E0472, // asm! is unsupported on this target
    E0473, // dereference of reference outside its lifetime
    E0474, // captured variable `..` does not outlive the enclosing closure
    E0475, // index of slice outside its lifetime
    E0476, // lifetime of the source pointer does not outlive lifetime bound...
    E0477, // the type `..` does not fulfill the required lifetime...
    E0478, // lifetime bound not satisfied
    E0479, // the type `..` (provided as the value of a type parameter) is...
    E0480, // lifetime of method receiver does not outlive the method call
    E0481, // lifetime of function argument does not outlive the function call
    E0482, // lifetime of return value does not outlive the function call
    E0483, // lifetime of operand does not outlive the operation
    E0484, // reference is not valid at the time of borrow
    E0485, // automatically reference is not valid at the time of borrow
    E0486, // type of expression contains references that are not valid during...
    E0487, // unsafe use of destructor: destructor might be called while...
    E0488, // lifetime of variable does not enclose its declaration
    E0489, // type/lifetime parameter not in scope here
    E0490, // a value of type `..` is borrowed for too long
    E0491, // in type `..`, reference has a longer lifetime than the data it...
    E0495, // cannot infer an appropriate lifetime due to conflicting requirements
}
