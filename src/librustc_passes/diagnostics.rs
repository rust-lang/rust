#![allow(non_snake_case)]

use syntax::{register_diagnostic, register_diagnostics, register_long_diagnostics};

register_long_diagnostics! {
/*
E0014: r##"
Constants can only be initialized by a constant value or, in a future
version of Rust, a call to a const function. This error indicates the use
of a path (like a::b, or x) denoting something other than one of these
allowed items. Erroneous code xample:

```compile_fail
const FOO: i32 = { let x = 0; x }; // 'x' isn't a constant nor a function!
```

To avoid it, you have to replace the non-constant value:

```
const FOO: i32 = { const X : i32 = 0; X };
// or even:
const FOO2: i32 = { 0 }; // but brackets are useless here
```
"##,
*/

E0130: r##"
You declared a pattern as an argument in a foreign function declaration.
Erroneous code example:

```compile_fail
extern {
    fn foo((a, b): (u32, u32)); // error: patterns aren't allowed in foreign
                                //        function declarations
}
```

Please replace the pattern argument with a regular one. Example:

```
struct SomeStruct {
    a: u32,
    b: u32,
}

extern {
    fn foo(s: SomeStruct); // ok!
}
```

Or:

```
extern {
    fn foo(a: (u32, u32)); // ok!
}
```
"##,

E0197: r##"
Inherent implementations (one that do not implement a trait but provide
methods associated with a type) are always safe because they are not
implementing an unsafe trait. Removing the `unsafe` keyword from the inherent
implementation will resolve this error.

```compile_fail,E0197
struct Foo;

// this will cause this error
unsafe impl Foo { }
// converting it to this will fix it
impl Foo { }
```
"##,

E0198: r##"
A negative implementation is one that excludes a type from implementing a
particular trait. Not being able to use a trait is always a safe operation,
so negative implementations are always safe and never need to be marked as
unsafe.

```compile_fail
#![feature(optin_builtin_traits)]

struct Foo;

// unsafe is unnecessary
unsafe impl !Clone for Foo { }
```

This will compile:

```ignore (ignore auto_trait future compatibility warning)
#![feature(optin_builtin_traits)]

struct Foo;

auto trait Enterprise {}

impl !Enterprise for Foo { }
```

Please note that negative impls are only allowed for auto traits.
"##,

E0267: r##"
This error indicates the use of a loop keyword (`break` or `continue`) inside a
closure but outside of any loop. Erroneous code example:

```compile_fail,E0267
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

```compile_fail,E0268
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

E0379: r##"
Trait methods cannot be declared `const` by design. For more information, see
[RFC 911].

[RFC 911]: https://github.com/rust-lang/rfcs/pull/911
"##,

E0380: r##"
Auto traits cannot have methods or associated items.
For more information see the [opt-in builtin traits RFC][RFC 19].

[RFC 19]: https://github.com/rust-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md
"##,

E0449: r##"
A visibility qualifier was used when it was unnecessary. Erroneous code
examples:

```compile_fail,E0449
struct Bar;

trait Foo {
    fn foo();
}

pub impl Bar {} // error: unnecessary visibility qualifier

pub impl Foo for Bar { // error: unnecessary visibility qualifier
    pub fn foo() {} // error: unnecessary visibility qualifier
}
```

To fix this error, please remove the visibility qualifier when it is not
required. Example:

```
struct Bar;

trait Foo {
    fn foo();
}

// Directly implemented methods share the visibility of the type itself,
// so `pub` is unnecessary here
impl Bar {}

// Trait methods share the visibility of the trait, so `pub` is
// unnecessary in either case
impl Foo for Bar {
    fn foo() {}
}
```
"##,


E0590: r##"
`break` or `continue` must include a label when used in the condition of a
`while` loop.

Example of erroneous code:

```compile_fail
while break {}
```

To fix this, add a label specifying which loop is being broken out of:
```
'foo: while break 'foo {}
```
"##,

E0571: r##"
A `break` statement with an argument appeared in a non-`loop` loop.

Example of erroneous code:

```compile_fail,E0571
# let mut i = 1;
# fn satisfied(n: usize) -> bool { n % 23 == 0 }
let result = while true {
    if satisfied(i) {
        break 2*i; // error: `break` with value from a `while` loop
    }
    i += 1;
};
```

The `break` statement can take an argument (which will be the value of the loop
expression if the `break` statement is executed) in `loop` loops, but not
`for`, `while`, or `while let` loops.

Make sure `break value;` statements only occur in `loop` loops:

```
# let mut i = 1;
# fn satisfied(n: usize) -> bool { n % 23 == 0 }
let result = loop { // ok!
    if satisfied(i) {
        break 2*i;
    }
    i += 1;
};
```
"##,

E0642: r##"
Trait methods currently cannot take patterns as arguments.

Example of erroneous code:

```compile_fail,E0642
trait Foo {
    fn foo((x, y): (i32, i32)); // error: patterns aren't allowed
                                //        in trait methods
}
```

You can instead use a single name for the argument:

```
trait Foo {
    fn foo(x_and_y: (i32, i32)); // ok!
}
```
"##,

E0695: r##"
A `break` statement without a label appeared inside a labeled block.

Example of erroneous code:

```compile_fail,E0695
# #![feature(label_break_value)]
loop {
    'a: {
        break;
    }
}
```

Make sure to always label the `break`:

```
# #![feature(label_break_value)]
'l: loop {
    'a: {
        break 'l;
    }
}
```

Or if you want to `break` the labeled block:

```
# #![feature(label_break_value)]
loop {
    'a: {
        break 'a;
    }
    break;
}
```
"##,

E0670: r##"
Rust 2015 does not permit the use of `async fn`.

Example of erroneous code:

```compile_fail,E0670
async fn foo() {}
```

Switch to the Rust 2018 edition to use `async fn`.
"##
}

register_diagnostics! {
    E0226, // only a single explicit lifetime bound is permitted
    E0472, // asm! is unsupported on this target
    E0561, // patterns aren't allowed in function pointer types
    E0567, // auto traits can not have generic parameters
    E0568, // auto traits can not have super traits
    E0666, // nested `impl Trait` is illegal
    E0667, // `impl Trait` in projections
    E0696, // `continue` pointing to a labeled block
    E0706, // `async fn` in trait
}
