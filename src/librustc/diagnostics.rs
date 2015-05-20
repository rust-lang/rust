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
"##,

E0002: r##"
This error indicates that an empty match expression is illegal because the type
it is matching on is non-empty (there exist values of this type). In safe code
it is impossible to create an instance of an empty type, so empty match
expressions are almost never desired.  This error is typically fixed by adding
one or more cases to the match expression.

An example of an empty type is `enum Empty { }`.
"##,

E0003: r##"
Not-a-Number (NaN) values cannot be compared for equality and hence can never
match the input to a match expression. To match against NaN values, you should
instead use the `is_nan` method in a guard, as in: `x if x.is_nan() => ...`
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

// FIXME: Remove duplication here?
E0005: r##"
Patterns used to bind names must be irrefutable, that is, they must guarantee
that a name will be extracted in all cases. If you encounter this error you
probably need to use a `match` or `if let` to deal with the possibility of
failure.
"##,

E0006: r##"
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
the heap at runtime, and therefore cannot be done at compile time.
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

E0015: r##"
The only function calls allowed in static or constant expressions are enum
variant constructors or struct constructors (for unit or tuple structs). This
is because Rust currently does not support compile-time function execution.
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
const X: u32 = 50;
const Y: *const u32 = &X;
println!("{:?}", Y);
```
"##,

E0020: r##"
This error indicates that an attempt was made to divide by zero (or take the
remainder of a zero divisor) in a static or constant expression.
"##,

E0079: r##"
Enum variants which contain no data can be given a custom integer
representation. This error indicates that the value provided is not an
integer literal and is therefore invalid.
"##,

E0080: r##"
This error indicates that the compiler was unable to sensibly evaluate an
integer expression provided as an enum discriminant. Attempting to divide by 0
or causing integer overflow are two ways to induce this error. For example:

```
enum Enum {
    X = (1 << 500),
    Y = (1 / 0)
}
```

Ensure that the expressions given can be evaluated as the desired integer type.
See the FFI section of the Reference for more information about using a custom
integer type:

http://doc.rust-lang.org/reference.html#ffi-attributes
"##,

E0133: r##"
Using unsafe functionality, such as dereferencing raw pointers and calling
functions via FFI or marked as unsafe, is potentially dangerous and disallowed
by safety checks. As such, those safety checks can be temporarily relaxed by
wrapping the unsafe instructions inside an `unsafe` block. For instance:

```
unsafe fn f() { return; }

fn main() {
    unsafe { f(); }
}
```

See also http://doc.rust-lang.org/book/unsafe.html
"##,

E0137: r##"
This error indicates that the compiler found multiple functions with the
`#[main]` attribute. This is an error because there must be a unique entry
point into a Rust program.
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
closure but outside of any loop. Break and continue can be used as normal inside
closures as long as they are also contained within a loop. To halt the execution
of a closure you should instead use a return statement.
"##,

E0268: r##"
This error indicates the use of a loop keyword (`break` or `continue`) outside
of a loop. Without a loop to break out of or continue in, no sensible action can
be taken.
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
// the constraint `<i8 as Trait>::AssociatedType=32`, and
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
// Code like this...
match Some(5) {
    ref op_num @ Some(num) => ...
    None => ...
}

// After.
match Some("hi".to_string()) {
    Some(ref s) => {
        let op_string_ref = &Some(&s);
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
variable. This error can occur for several cases, the most common of which is a
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
"##

}


register_diagnostics! {
    E0011,
    E0014,
    E0016,
    E0017,
    E0019,
    E0022,
    E0038,
    E0109,
    E0110,
    E0134,
    E0135,
    E0136,
    E0138,
    E0139,
    E0261, // use of undeclared lifetime name
    E0262, // illegal lifetime parameter name
    E0263, // lifetime name declared twice in same scope
    E0264, // unknown external lang item
    E0266, // expected item
    E0269, // not all control paths return a value
    E0270, // computation may converge in a function marked as diverging
    E0272, // rustc_on_unimplemented attribute refers to non-existent type parameter
    E0273, // rustc_on_unimplemented must have named format arguments
    E0274, // rustc_on_unimplemented must have a value
    E0275, // overflow evaluating requirement
    E0276, // requirement appears on impl method but not on corresponding trait method
    E0277, // trait is not implemented for type
    E0278, // requirement is not satisfied
    E0279, // requirement is not satisfied
    E0280, // requirement is not satisfied
    E0281, // type implements trait but other trait is required
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
    E0370  // discriminant overflow
}
