// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

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
E0030: r##"
When matching against a range, the compiler verifies that the range is
non-empty.  Range patterns include both end-points, so this is equivalent to
requiring the start of the range to be less than or equal to the end of the
range.

For example:

```compile_fail
match 5u32 {
    // This range is ok, albeit pointless.
    1 ... 1 => {}
    // This range is empty, and the compiler can tell.
    1000 ... 5 => {}
}
```
"##,

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

E0161: r##"
A value was moved. However, its size was not known at compile time, and only
values of a known size can be moved.

Erroneous code example:

```compile_fail
#![feature(box_syntax)]

fn main() {
    let array: &[isize] = &[1, 2, 3];
    let _x: Box<[isize]> = box *array;
    // error: cannot move a value of type [isize]: the size of [isize] cannot
    //        be statically determined
}
```

In Rust, you can only move a value when its size is known at compile time.

To work around this restriction, consider "hiding" the value behind a reference:
either `&x` or `&mut x`. Since a reference has a fixed size, this lets you move
it around as usual. Example:

```
#![feature(box_syntax)]

fn main() {
    let array: &[isize] = &[1, 2, 3];
    let _x: Box<&[isize]> = box array; // ok!
}
```
"##,

E0265: r##"
This error indicates that a static or constant references itself.
All statics and constants need to resolve to a value in an acyclic manner.

For example, neither of the following can be sensibly compiled:

```compile_fail,E0265
const X: u32 = X;
```

```compile_fail,E0265
const X: u32 = Y;
const Y: u32 = X;
```
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

E0449: r##"
A visibility qualifier was used when it was unnecessary. Erroneous code
examples:

```compile_fail
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

```ignore
struct Bar;

trait Foo {
    fn foo();
}

// Directly implemented methods share the visibility of the type itself,
// so `pub` is unnecessary here
impl Bar {}

// Trait methods share the visibility of the trait, so `pub` is
// unnecessary in either case
pub impl Foo for Bar {
    pub fn foo() {}
}
```
"##,

}

register_diagnostics! {
    E0472, // asm! is unsupported on this target
    E0561, // patterns aren't allowed in function pointer types
    E0571, // `break` with a value in a non-`loop`-loop
}
