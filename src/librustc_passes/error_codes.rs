syntax::register_diagnostics! {
E0014: r##"
#### Note: this error code is no longer emitted by the compiler.

Constants can only be initialized by a constant value or, in a future
version of Rust, a call to a const function. This error indicates the use
of a path (like a::b, or x) denoting something other than one of these
allowed items.

Erroneous code example:

```
const FOO: i32 = { let x = 0; x }; // 'x' isn't a constant nor a function!
```

To avoid it, you have to replace the non-constant value:

```
const FOO: i32 = { const X : i32 = 0; X };
// or even:
const FOO2: i32 = { 0 }; // but brackets are useless here
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

// This shouldn't really ever trigger since the repeated value error comes first
E0136: r##"
A binary can only have one entry point, and by default that entry point is the
function `main()`. If there are multiple such functions, please rename one.

Erroneous code example:

```compile_fail,E0136
fn main() {
    // ...
}

// ...

fn main() { // error!
    // ...
}
```
"##,

E0137: r##"
More than one function was declared with the `#[main]` attribute.

Erroneous code example:

```compile_fail,E0137
#![feature(main)]

#[main]
fn foo() {}

#[main]
fn f() {} // error: multiple functions with a `#[main]` attribute
```

This error indicates that the compiler found multiple functions with the
`#[main]` attribute. This is an error because there must be a unique entry
point into a Rust program. Example:

```
#![feature(main)]

#[main]
fn f() {} // ok!
```
"##,

E0138: r##"
More than one function was declared with the `#[start]` attribute.

Erroneous code example:

```compile_fail,E0138
#![feature(start)]

#[start]
fn foo(argc: isize, argv: *const *const u8) -> isize {}

#[start]
fn f(argc: isize, argv: *const *const u8) -> isize {}
// error: multiple 'start' functions
```

This error indicates that the compiler found multiple functions with the
`#[start]` attribute. This is an error because there must be a unique entry
point into a Rust program. Example:

```
#![feature(start)]

#[start]
fn foo(argc: isize, argv: *const *const u8) -> isize { 0 } // ok!
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
    break; // error: `break` outside of a loop
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

E0512: r##"
Transmute with two differently sized types was attempted. Erroneous code
example:

```compile_fail,E0512
fn takes_u8(_: u8) {}

fn main() {
    unsafe { takes_u8(::std::mem::transmute(0u16)); }
    // error: cannot transmute between types of different sizes,
    //        or dependently-sized types
}
```

Please use types with same size or use the expected type directly. Example:

```
fn takes_u8(_: u8) {}

fn main() {
    unsafe { takes_u8(::std::mem::transmute(0i8)); } // ok!
    // or:
    unsafe { takes_u8(0u8); } // ok!
}
```
"##,

E0561: r##"
A non-ident or non-wildcard pattern has been used as a parameter of a function
pointer type.

Erroneous code example:

```compile_fail,E0561
type A1 = fn(mut param: u8); // error!
type A2 = fn(&param: u32); // error!
```

When using an alias over a function type, you cannot e.g. denote a parameter as
being mutable.

To fix the issue, remove patterns (`_` is allowed though). Example:

```
type A1 = fn(param: u8); // ok!
type A2 = fn(_: u32); // ok!
```

You can also omit the parameter name:

```
type A3 = fn(i16); // ok!
```
"##,

E0567: r##"
Generics have been used on an auto trait.

Erroneous code example:

```compile_fail,E0567
#![feature(optin_builtin_traits)]

auto trait Generic<T> {} // error!

fn main() {}
```

Since an auto trait is implemented on all existing types, the
compiler would not be able to infer the types of the trait's generic
parameters.

To fix this issue, just remove the generics:

```
#![feature(optin_builtin_traits)]

auto trait Generic {} // ok!

fn main() {}
```
"##,

E0568: r##"
A super trait has been added to an auto trait.

Erroneous code example:

```compile_fail,E0568
#![feature(optin_builtin_traits)]

auto trait Bound : Copy {} // error!

fn main() {}
```

Since an auto trait is implemented on all existing types, adding a super trait
would filter out a lot of those types. In the current example, almost none of
all the existing types could implement `Bound` because very few of them have the
`Copy` trait.

To fix this issue, just remove the super trait:

```
#![feature(optin_builtin_traits)]

auto trait Bound {} // ok!

fn main() {}
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

E0591: r##"
Per [RFC 401][rfc401], if you have a function declaration `foo`:

```
// For the purposes of this explanation, all of these
// different kinds of `fn` declarations are equivalent:
struct S;
fn foo(x: S) { /* ... */ }
# #[cfg(for_demonstration_only)]
extern "C" { fn foo(x: S); }
# #[cfg(for_demonstration_only)]
impl S { fn foo(self) { /* ... */ } }
```

the type of `foo` is **not** `fn(S)`, as one might expect.
Rather, it is a unique, zero-sized marker type written here as `typeof(foo)`.
However, `typeof(foo)` can be _coerced_ to a function pointer `fn(S)`,
so you rarely notice this:

```
# struct S;
# fn foo(_: S) {}
let x: fn(S) = foo; // OK, coerces
```

The reason that this matter is that the type `fn(S)` is not specific to
any particular function: it's a function _pointer_. So calling `x()` results
in a virtual call, whereas `foo()` is statically dispatched, because the type
of `foo` tells us precisely what function is being called.

As noted above, coercions mean that most code doesn't have to be
concerned with this distinction. However, you can tell the difference
when using **transmute** to convert a fn item into a fn pointer.

This is sometimes done as part of an FFI:

```compile_fail,E0591
extern "C" fn foo(userdata: Box<i32>) {
    /* ... */
}

# fn callback(_: extern "C" fn(*mut i32)) {}
# use std::mem::transmute;
# unsafe {
let f: extern "C" fn(*mut i32) = transmute(foo);
callback(f);
# }
```

Here, transmute is being used to convert the types of the fn arguments.
This pattern is incorrect because, because the type of `foo` is a function
**item** (`typeof(foo)`), which is zero-sized, and the target type (`fn()`)
is a function pointer, which is not zero-sized.
This pattern should be rewritten. There are a few possible ways to do this:

- change the original fn declaration to match the expected signature,
  and do the cast in the fn body (the preferred option)
- cast the fn item fo a fn pointer before calling transmute, as shown here:

    ```
    # extern "C" fn foo(_: Box<i32>) {}
    # use std::mem::transmute;
    # unsafe {
    let f: extern "C" fn(*mut i32) = transmute(foo as extern "C" fn(_));
    let f: extern "C" fn(*mut i32) = transmute(foo as usize); // works too
    # }
    ```

The same applies to transmutes to `*mut fn()`, which were observed in practice.
Note though that use of this type is generally incorrect.
The intention is typically to describe a function pointer, but just `fn()`
alone suffices for that. `*mut fn()` is a pointer to a fn pointer.
(Since these values are typically just passed to C code, however, this rarely
makes a difference in practice.)

[rfc401]: https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md
"##,

E0601: r##"
No `main` function was found in a binary crate. To fix this error, add a
`main` function. For example:

```
fn main() {
    // Your program will start here.
    println!("Hello world!");
}
```

If you don't know the basics of Rust, you can go look to the Rust Book to get
started: https://doc.rust-lang.org/book/
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

E0666: r##"
`impl Trait` types cannot appear nested in the
generic arguments of other `impl Trait` types.

Example of erroneous code:

```compile_fail,E0666
trait MyGenericTrait<T> {}
trait MyInnerTrait {}

fn foo(bar: impl MyGenericTrait<impl MyInnerTrait>) {}
```

Type parameters for `impl Trait` types must be
explicitly defined as named generic parameters:

```
trait MyGenericTrait<T> {}
trait MyInnerTrait {}

fn foo<T: MyInnerTrait>(bar: impl MyGenericTrait<T>) {}
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
"##,

;
    E0226, // only a single explicit lifetime bound is permitted
    E0472, // asm! is unsupported on this target
    E0667, // `impl Trait` in projections
    E0696, // `continue` pointing to a labeled block
    E0706, // `async fn` in trait
}
