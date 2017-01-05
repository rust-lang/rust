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
## Note: this error code is no longer emitted by the compiler.

This error suggests that the expression arm corresponding to the noted pattern
will never be reached as for all possible values of the expression being
matched, one of the preceding patterns will match.

This means that perhaps some of the preceding patterns are too general, this
one is too specific or the ordering is incorrect.

For example, the following `match` block has too many arms:

```
match Some(0) {
    Some(bar) => {/* ... */}
    x => {/* ... */} // This handles the `None` case
    _ => {/* ... */} // All possible cases have already been handled
}
```

`match` blocks have their patterns matched in order, so, for example, putting
a wildcard arm above a more specific arm will make the latter arm irrelevant.

Ensure the ordering of the match arm is correct and remove any superfluous
arms.
"##,

E0002: r##"
## Note: this error code is no longer emitted by the compiler.

This error indicates that an empty match expression is invalid because the type
it is matching on is non-empty (there exist values of this type). In safe code
it is impossible to create an instance of an empty type, so empty match
expressions are almost never desired. This error is typically fixed by adding
one or more cases to the match expression.

An example of an empty type is `enum Empty { }`. So, the following will work:

```
enum Empty {}

fn foo(x: Empty) {
    match x {
        // empty
    }
}
```

However, this won't:

```compile_fail
fn foo(x: Option<String>) {
    match x {
        // empty
    }
}
```
"##,

E0003: r##"
## Note: this error code is no longer emitted by the compiler.

Not-a-Number (NaN) values cannot be compared for equality and hence can never
match the input to a match expression. So, the following will not compile:

```compile_fail
const NAN: f32 = 0.0 / 0.0;

let number = 0.1f32;

match number {
    NAN => { /* ... */ },
    _ => {}
}
```

To match against NaN values, you should instead use the `is_nan()` method in a
guard, like so:

```
let number = 0.1f32;

match number {
    x if x.is_nan() => { /* ... */ }
    _ => {}
}
```
"##,

E0004: r##"
This error indicates that the compiler cannot guarantee a matching pattern for
one or more possible inputs to a match expression. Guaranteed matches are
required in order to assign values to match expressions, or alternatively,
determine the flow of execution. Erroneous code example:

```compile_fail,E0004
enum Terminator {
    HastaLaVistaBaby,
    TalkToMyHand,
}

let x = Terminator::HastaLaVistaBaby;

match x { // error: non-exhaustive patterns: `HastaLaVistaBaby` not covered
    Terminator::TalkToMyHand => {}
}
```

If you encounter this error you must alter your patterns so that every possible
value of the input type is matched. For types with a small number of variants
(like enums) you should probably cover all cases explicitly. Alternatively, the
underscore `_` wildcard pattern can be added after all other patterns to match
"anything else". Example:

```
enum Terminator {
    HastaLaVistaBaby,
    TalkToMyHand,
}

let x = Terminator::HastaLaVistaBaby;

match x {
    Terminator::TalkToMyHand => {}
    Terminator::HastaLaVistaBaby => {}
}

// or:

match x {
    Terminator::TalkToMyHand => {}
    _ => {}
}
```
"##,

E0005: r##"
Patterns used to bind names must be irrefutable, that is, they must guarantee
that a name will be extracted in all cases. Erroneous code example:

```compile_fail,E0005
let x = Some(1);
let Some(y) = x;
// error: refutable pattern in local binding: `None` not covered
```

If you encounter this error you probably need to use a `match` or `if let` to
deal with the possibility of failure. Example:

```
let x = Some(1);

match x {
    Some(y) => {
        // do something
    },
    None => {}
}

// or:

if let Some(y) = x {
    // do something
}
```
"##,

E0007: r##"
This error indicates that the bindings in a match arm would require a value to
be moved into more than one location, thus violating unique ownership. Code
like the following is invalid as it requires the entire `Option<String>` to be
moved into a variable called `op_string` while simultaneously requiring the
inner `String` to be moved into a variable called `s`.

```compile_fail,E0007
let x = Some("s".to_string());

match x {
    op_string @ Some(s) => {}, // error: cannot bind by-move with sub-bindings
    None => {},
}
```

See also the error E0303.
"##,

E0008: r##"
Names bound in match arms retain their type in pattern guards. As such, if a
name is bound by move in a pattern, it should also be moved to wherever it is
referenced in the pattern guard code. Doing so however would prevent the name
from being available in the body of the match arm. Consider the following:

```compile_fail,E0008
match Some("hi".to_string()) {
    Some(s) if s.len() == 0 => {}, // use s.
    _ => {},
}
```

The variable `s` has type `String`, and its use in the guard is as a variable of
type `String`. The guard code effectively executes in a separate scope to the
body of the arm, so the value would be moved into this anonymous scope and
therefore becomes unavailable in the body of the arm.

The problem above can be solved by using the `ref` keyword.

```
match Some("hi".to_string()) {
    Some(ref s) if s.len() == 0 => {},
    _ => {},
}
```

Though this example seems innocuous and easy to solve, the problem becomes clear
when it encounters functions which consume the value:

```compile_fail,E0008
struct A{}

impl A {
    fn consume(self) -> usize {
        0
    }
}

fn main() {
    let a = Some(A{});
    match a {
        Some(y) if y.consume() > 0 => {}
        _ => {}
    }
}
```

In this situation, even the `ref` keyword cannot solve it, since borrowed
content cannot be moved. This problem cannot be solved generally. If the value
can be cloned, here is a not-so-specific solution:

```
#[derive(Clone)]
struct A{}

impl A {
    fn consume(self) -> usize {
        0
    }
}

fn main() {
    let a = Some(A{});
    match a{
        Some(ref y) if y.clone().consume() > 0 => {}
        _ => {}
    }
}
```

If the value will be consumed in the pattern guard, using its clone will not
move its ownership, so the code works.
"##,

E0009: r##"
In a pattern, all values that don't implement the `Copy` trait have to be bound
the same way. The goal here is to avoid binding simultaneously by-move and
by-ref.

This limitation may be removed in a future version of Rust.

Erroneous code example:

```compile_fail,E0009
struct X { x: (), }

let x = Some((X { x: () }, X { x: () }));
match x {
    Some((y, ref z)) => {}, // error: cannot bind by-move and by-ref in the
                            //        same pattern
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
    Some(x) if x == FORTY_TWO => {}
    _ => {}
}
```
"##,

E0162: r##"
An if-let pattern attempts to match the pattern, and enters the body if the
match was successful. If the match is irrefutable (when it cannot fail to
match), use a regular `let`-binding instead. For instance:

```compile_fail,E0162
struct Irrefutable(i32);
let irr = Irrefutable(0);

// This fails to compile because the match is irrefutable.
if let Irrefutable(x) = irr {
    // This body will always be executed.
    // ...
}
```

Try this instead:

```
struct Irrefutable(i32);
let irr = Irrefutable(0);

let Irrefutable(x) = irr;
println!("{}", x);
```
"##,

E0165: r##"
A while-let pattern attempts to match the pattern, and enters the body if the
match was successful. If the match is irrefutable (when it cannot fail to
match), use a regular `let`-binding inside a `loop` instead. For instance:

```compile_fail,E0165
struct Irrefutable(i32);
let irr = Irrefutable(0);

// This fails to compile because the match is irrefutable.
while let Irrefutable(x) = irr {
    // ...
}
```

Try this instead:

```no_run
struct Irrefutable(i32);
let irr = Irrefutable(0);

loop {
    let Irrefutable(x) = irr;
    // ...
}
```
"##,

E0170: r##"
Enum variants are qualified by default. For example, given this type:

```
enum Method {
    GET,
    POST,
}
```

You would match it using:

```
enum Method {
    GET,
    POST,
}

let m = Method::GET;

match m {
    Method::GET => {},
    Method::POST => {},
}
```

If you don't qualify the names, the code will bind new variables named "GET" and
"POST" instead. This behavior is likely not what you want, so `rustc` warns when
that happens.

Qualified names are good practice, and most code works well with them. But if
you prefer them unqualified, you can import the variants into scope:

```ignore
use Method::*;
enum Method { GET, POST }
```

If you want others to be able to import variants from your module directly, use
`pub use`:

```ignore
pub use Method::*;
enum Method { GET, POST }
```
"##,


E0297: r##"
Patterns used to bind names must be irrefutable. That is, they must guarantee
that a name will be extracted in all cases. Instead of pattern matching the
loop variable, consider using a `match` or `if let` inside the loop body. For
instance:

```compile_fail,E0297
let xs : Vec<Option<i32>> = vec![Some(1), None];

// This fails because `None` is not covered.
for Some(x) in xs {
    // ...
}
```

Match inside the loop instead:

```
let xs : Vec<Option<i32>> = vec![Some(1), None];

for item in xs {
    match item {
        Some(x) => {},
        None => {},
    }
}
```

Or use `if let`:

```
let xs : Vec<Option<i32>> = vec![Some(1), None];

for item in xs {
    if let Some(x) = item {
        // ...
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

```compile_fail,E0301
match Some(()) {
    None => { },
    option if option.take().is_none() => {
        /* impossible, option is `Some` */
    },
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

```compile_fail,E0302
match Some(()) {
    None => { },
    option if { option = None; false } => { },
    Some(_) => { } // When the previous match failed, the option became `None`.
}
```
"##,

E0303: r##"
In certain cases it is possible for sub-bindings to violate memory safety.
Updates to the borrow checker in a future version of Rust may remove this
restriction, but for now patterns must be rewritten without sub-bindings.

Before:

```compile_fail,E0303
match Some("hi".to_string()) {
    ref op_string_ref @ Some(s) => {},
    None => {},
}
```

After:

```
match Some("hi".to_string()) {
    Some(ref s) => {
        let op_string_ref = &Some(s);
        // ...
    },
    None => {},
}
```

The `op_string_ref` binding has type `&Option<&String>` in both cases.

See also https://github.com/rust-lang/rust/issues/14587
"##,

E0080: r##"
This error indicates that the compiler was unable to sensibly evaluate an
constant expression that had to be evaluated. Attempting to divide by 0
or causing integer overflow are two ways to induce this error. For example:

```compile_fail,E0080
enum Enum {
    X = (1 << 500),
    Y = (1 / 0)
}
```

Ensure that the expressions given can be evaluated as the desired integer type.
See the FFI section of the Reference for more information about using a custom
integer type:

https://doc.rust-lang.org/reference.html#ffi-attributes
"##,


E0306: r##"
In an array type `[T; N]`, `N` is the number of elements in the array. This
must be an unsigned integer. Erroneous code example:

```compile_fail,E0306
const X: [i32; true] = [0]; // error: expected `usize` for array length,
                            //        found boolean
```

Working example:

```
const X: [i32; 1] = [0];
```
"##,
}


register_diagnostics! {
    E0298, // cannot compare constants
//  E0299, // mismatched types between arms
//  E0471, // constant evaluation error (in pattern)
}
