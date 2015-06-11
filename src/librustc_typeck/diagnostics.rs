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

register_long_diagnostics! {

E0023: r##"
A pattern used to match against an enum variant must provide a sub-pattern for
each field of the enum variant. This error indicates that a pattern attempted to
extract an incorrect number of fields from a variant.

```
enum Fruit {
    Apple(String, String)
    Pear(u32)
}
```

Here the `Apple` variant has two fields, and should be matched against like so:

```
// Correct.
match x {
    Apple(a, b) => ...
}
```

Matching with the wrong number of fields has no sensible interpretation:

```
// Incorrect.
match x {
    Apple(a) => ...,
    Apple(a, b, c) => ...
}
```

Check how many fields the enum was declared with and ensure that your pattern
uses the same number.
"##,

E0024: r##"
This error indicates that a pattern attempted to extract the fields of an enum
variant with no fields. Here's a tiny example of this error:

```
// This enum has two variants.
enum Number {
    // This variant has no fields.
    Zero,
    // This variant has one field.
    One(u32)
}

// Assuming x is a Number we can pattern match on its contents.
match x {
    Zero(inside) => ...,
    One(inside) => ...
}
```

The pattern match `Zero(inside)` is incorrect because the `Zero` variant
contains no fields, yet the `inside` name attempts to bind the first field of
the enum.
"##,

E0025: r##"
Each field of a struct can only be bound once in a pattern. Each occurrence of a
field name binds the value of that field, so to fix this error you will have to
remove or alter the duplicate uses of the field name. Perhaps you misspelt
another field name?
"##,

E0026: r##"
This error indicates that a struct pattern attempted to extract a non-existant
field from a struct. Struct fields are identified by the name used before the
colon `:` so struct patterns should resemble the declaration of the struct type
being matched.

```
// Correct matching.
struct Thing {
    x: u32,
    y: u32
}

let thing = Thing { x: 1, y: 2 };
match thing {
    Thing { x: xfield, y: yfield } => ...
}
```

If you are using shorthand field patterns but want to refer to the struct field
by a different name, you should rename it explicitly.

```
// Change this:
match thing {
    Thing { x, z } => ...
}

// To this:
match thing {
    Thing { x, y: z } => ...
}
```
"##,

E0027: r##"
This error indicates that a pattern for a struct fails to specify a sub-pattern
for every one of the struct's fields. Ensure that each field from the struct's
definition is mentioned in the pattern, or use `..` to ignore unwanted fields.

For example:

```
struct Dog {
    name: String,
    age: u32
}

let d = Dog { name: "Rusty".to_string(), age: 8 };

// This is incorrect.
match d {
    Dog { age: x } => ...
}

// This is correct (explicit).
match d {
    Dog { name: n, age: x } => ...
}

// This is also correct (ignore unused fields).
match d {
    Dog { age: x, .. } => ...
}
```
"##,

E0029: r##"
In a match expression, only numbers and characters can be matched against a
range. This is because the compiler checks that the range is non-empty at
compile-time, and is unable to evaluate arbitrary comparison functions. If you
want to capture values of an orderable type between two end-points, you can use
a guard.

```
// The ordering relation for strings can't be evaluated at compile time,
// so this doesn't work:
match string {
    "hello" ... "world" => ...
    _ => ...
}

// This is a more general version, using a guard:
match string {
    s if s >= "hello" && s <= "world" => ...
    _ => ...
}
```
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

E0033: r##"
This error indicates that a pointer to a trait type cannot be implicitly
dereferenced by a pattern. Every trait defines a type, but because the
size of trait implementors isn't fixed, this type has no compile-time size.
Therefore, all accesses to trait types must be through pointers. If you
encounter this error you should try to avoid dereferencing the pointer.

```
let trait_obj: &SomeTrait = ...;

// This tries to implicitly dereference to create an unsized local variable.
let &invalid = trait_obj;

// You can call methods without binding to the value being pointed at.
trait_obj.method_one();
trait_obj.method_two();
```

You can read more about trait objects in the Trait Object section of the
Reference:

http://doc.rust-lang.org/reference.html#trait-objects
"##,

E0040: r##"
It is not allowed to manually call destructors in Rust. It is also not
necessary to do this since `drop` is called automatically whenever a value goes
out of scope.

Here's an example of this error:

```
struct Foo {
    x: i32,
}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("kaboom");
    }
}

fn main() {
    let mut x = Foo { x: -7 };
    x.drop(); // error: explicit use of destructor method
}
```
"##,

E0046: r##"
When trying to make some type implement a trait `Foo`, you must, at minimum,
provide implementations for all of `Foo`'s required methods (meaning the
methods that do not have default implementations), as well as any required
trait items like associated types or constants.
"##,

E0049: r##"
This error indicates that an attempted implementation of a trait method
has the wrong number of type parameters.

For example, the trait below has a method `foo` with a type parameter `T`,
but the implementation of `foo` for the type `Bar` is missing this parameter:

```
trait Foo {
    fn foo<T: Default>(x: T) -> Self;
}

struct Bar;

// error: method `foo` has 0 type parameters but its trait declaration has 1
// type parameter
impl Foo for Bar {
    fn foo(x: bool) -> Self { Bar }
}
```
"##,

E0050: r##"
This error indicates that an attempted implementation of a trait method
has the wrong number of function parameters.

For example, the trait below has a method `foo` with two function parameters
(`&self` and `u8`), but the implementation of `foo` for the type `Bar` omits
the `u8` parameter:

```
trait Foo {
    fn foo(&self, x: u8) -> bool;
}

struct Bar;

// error: method `foo` has 1 parameter but the declaration in trait `Foo::foo`
// has 2
impl Foo for Bar {
    fn foo(&self) -> bool { true }
}
```
"##,

E0053: r##"
The parameters of any trait method must match between a trait implementation
and the trait definition.

Here are a couple examples of this error:

```
trait Foo {
    fn foo(x: u16);
    fn bar(&self);
}

struct Bar;

impl Foo for Bar {
    // error, expected u16, found i16
    fn foo(x: i16) { }

    // error, values differ in mutability
    fn bar(&mut self) { }
}
```
"##,

E0054: r##"
It is not allowed to cast to a bool. If you are trying to cast a numeric type
to a bool, you can compare it with zero instead:

```
let x = 5;

// Ok
let x_is_nonzero = x != 0;

// Not allowed, won't compile
let x_is_nonzero = x as bool;
```
"##,

E0055: r##"
During a method call, a value is automatically dereferenced as many times as
needed to make the value's type match the method's receiver. The catch is that
the compiler will only attempt to dereference a number of times up to the
recursion limit (which can be set via the `recursion_limit` attribute).

For a somewhat artificial example:

```
#![recursion_limit="2"]

struct Foo;

impl Foo {
    fn foo(&self) {}
}

fn main() {
    let foo = Foo;
    let ref_foo = &&Foo;

    // error, reached the recursion limit while auto-dereferencing &&Foo
    ref_foo.foo();
}
```

One fix may be to increase the recursion limit. Note that it is possible to
create an infinite recursion of dereferencing, in which case the only fix is to
somehow break the recursion.
"##,

E0057: r##"
When invoking closures or other implementations of the function traits `Fn`,
`FnMut` or `FnOnce` using call notation, the number of parameters passed to the
function must match its definition.

An example using a closure:

```
let f = |x| x * 3;
let a = f();        // invalid, too few parameters
let b = f(4);       // this works!
let c = f(2, 3);    // invalid, too many parameters
```

A generic function must be treated similarly:

```
fn foo<F: Fn()>(f: F) {
    f(); // this is valid, but f(3) would not work
}
```
"##,

E0059: r##"
The built-in function traits are generic over a tuple of the function arguments.
If one uses angle-bracket notation (`Fn<(T,), Output=U>`) instead of parentheses
(`Fn(T) -> U`) to denote the function trait, the type parameter should be a
tuple. Otherwise function call notation cannot be used and the trait will not be
implemented by closures.

The most likely source of this error is using angle-bracket notation without
wrapping the function argument type into a tuple, for example:

```
fn foo<F: Fn<i32>>(f: F) -> F::Output { f(3) }
```

It can be fixed by adjusting the trait bound like this:

```
fn foo<F: Fn<(i32,)>>(f: F) -> F::Output { f(3) }
```

Note that `(T,)` always denotes the type of a 1-tuple containing an element of
type `T`. The comma is necessary for syntactic disambiguation.
"##,

E0060: r##"
External C functions are allowed to be variadic. However, a variadic function
takes a minimum number of arguments. For example, consider C's variadic `printf`
function:

```
extern crate libc;
use libc::{ c_char, c_int };

extern "C" {
    fn printf(_: *const c_char, ...) -> c_int;
}
```

Using this declaration, it must be called with at least one argument, so
simply calling `printf()` is illegal. But the following uses are allowed:

```
unsafe {
    use std::ffi::CString;

    printf(CString::new("test\n").unwrap().as_ptr());
    printf(CString::new("number = %d\n").unwrap().as_ptr(), 3);
    printf(CString::new("%d, %d\n").unwrap().as_ptr(), 10, 5);
}
```
"##,

E0061: r##"
The number of arguments passed to a function must match the number of arguments
specified in the function signature.

For example, a function like

```
fn f(a: u16, b: &str) {}
```

must always be called with exactly two arguments, e.g. `f(2, "test")`.

Note, that Rust does not have a notion of optional function arguments or
variadic functions (except for its C-FFI).
"##,

E0062: r##"
This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was specified more than once. Each field should
be specified exactly one time.
"##,

E0063: r##"
This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was not provided. Each field should be
specified exactly once.
"##,

E0066: r##"
Box placement expressions (like C++'s "placement new") do not yet support any
place expression except the exchange heap (i.e. `std::boxed::HEAP`).
Furthermore, the syntax is changing to use `in` instead of `box`. See [RFC 470]
and [RFC 809] for more details.

[RFC 470]: https://github.com/rust-lang/rfcs/pull/470
[RFC 809]: https://github.com/rust-lang/rfcs/pull/809
"##,

E0067: r##"
The left-hand side of a compound assignment expression must be an lvalue
expression. An lvalue expression represents a memory location and includes
item paths (ie, namespaced variables), dereferences, indexing expressions,
and field references.

Let's start with some bad examples:
```
use std::collections::LinkedList;

// Bad: assignment to non-lvalue expression
LinkedList::new() += 1;

// ...

fn some_func(i: &mut i32) {
    i += 12; // Error : '+=' operation cannot be applied on a reference !
}

And now some good examples:
```
let mut i : i32 = 0;

i += 12; // Good !

// ...

fn some_func(i: &mut i32) {
    *i += 12; // Good !
}

```
"##,

E0069: r##"
The compiler found a function whose body contains a `return;` statement but
whose return type is not `()`. An example of this is:

```
// error
fn foo() -> u8 {
    return;
}
```

Since `return;` is just like `return ();`, there is a mismatch between the
function's return type and the value being returned.
"##,

E0070: r##"
The left-hand side of an assignment operator must be an lvalue expression. An
lvalue expression represents a memory location and can be a variable (with
optional namespacing), a dereference, an indexing expression or a field
reference.

More details can be found here:
https://doc.rust-lang.org/reference.html#lvalues,-rvalues-and-temporaries

Now, we can go further. Here are some bad examples:
```
struct SomeStruct {
    x: i32,
    y: i32
}
const SOME_CONST : i32 = 12;

fn some_other_func() {}

fn some_function() {
    SOME_CONST = 14; // error : a constant value cannot be changed!
    1 = 3; // error : 1 isn't a valid lvalue!
    some_other_func() = 4; // error : we can't assign value to a function!
    SomeStruct.x = 12; // error : SomeStruct a structure name but it is used
                       // like a variable!
}
```

And now let's give good examples:

```
struct SomeStruct {
    x: i32,
    y: i32
}
let mut s = SomeStruct {x: 0, y: 0};

s.x = 3; // that's good !

// ...

fn some_func(x: &mut i32) {
    *x = 12; // that's good !
}
```
"##,

E0072: r##"
When defining a recursive struct or enum, any use of the type being defined
from inside the definition must occur behind a pointer (like `Box` or `&`).
This is because structs and enums must have a well-defined size, and without
the pointer the size of the type would need to be unbounded.

Consider the following erroneous definition of a type for a list of bytes:

```
// error, illegal recursive struct type
struct ListNode {
    head: u8,
    tail: Option<ListNode>,
}
```

This type cannot have a well-defined size, because it needs to be arbitrarily
large (since we would be able to nest `ListNode`s to any depth). Specifically,

```
size of ListNode = 1 byte for head
                 + 1 byte for the discriminant of the Option
                 + size of ListNode
```

One way to fix this is by wrapping `ListNode` in a `Box`, like so:

```
struct ListNode {
    head: u8,
    tail: Option<Box<ListNode>>,
}
```

This works because `Box` is a pointer, so its size is well-known.
"##,

E0073: r##"
You cannot define a struct (or enum) `Foo` that requires an instance of `Foo`
in order to make a new `Foo` value. This is because there would be no way a
first instance of `Foo` could be made to initialize another instance!

Here's an example of a struct that has this problem:

```
struct Foo { x: Box<Foo> } // error
```

One fix is to use `Option`, like so:

```
struct Foo { x: Option<Box<Foo>> }
```

Now it's possible to create at least one instance of `Foo`: `Foo { x: None }`.
"##,

E0081: r##"
Enum discriminants are used to differentiate enum variants stored in memory.
This error indicates that the same value was used for two or more variants,
making them impossible to tell apart.

```
// Good.
enum Enum {
    P,
    X = 3,
    Y = 5
}

// Bad.
enum Enum {
    P = 3,
    X = 3,
    Y = 5
}
```

Note that variants without a manually specified discriminant are numbered from
top to bottom starting from 0, so clashes can occur with seemingly unrelated
variants.

```
enum Bad {
    X,
    Y = 0
}
```

Here `X` will have already been assigned the discriminant 0 by the time `Y` is
encountered, so a conflict occurs.
"##,

E0082: r##"
The default type for enum discriminants is `isize`, but it can be adjusted by
adding the `repr` attribute to the enum declaration. This error indicates that
an integer literal given as a discriminant is not a member of the discriminant
type. For example:

```
#[repr(u8)]
enum Thing {
    A = 1024,
    B = 5
}
```

Here, 1024 lies outside the valid range for `u8`, so the discriminant for `A` is
invalid. You may want to change representation types to fix this, or else change
invalid discriminant values so that they fit within the existing type.

Note also that without a representation manually defined, the compiler will
optimize by using the smallest integer type possible.
"##,

E0083: r##"
At present, it's not possible to define a custom representation for an enum with
a single variant. As a workaround you can add a `Dummy` variant.

See: https://github.com/rust-lang/rust/issues/10292
"##,

E0084: r##"
It is impossible to define an integer type to be used to represent zero-variant
enum values because there are no zero-variant enum values. There is no way to
construct an instance of the following type using only safe code:

```
enum Empty {}
```
"##,

E0087: r##"
Too many type parameters were supplied for a function. For example:

```
fn foo<T>() {}

fn main() {
    foo::<f64, bool>(); // error, expected 1 parameter, found 2 parameters
}
```

The number of supplied parameters much exactly match the number of defined type
parameters.
"##,

E0089: r##"
Not enough type parameters were supplied for a function. For example:

```
fn foo<T, U>() {}

fn main() {
    foo::<f64>(); // error, expected 2 parameters, found 1 parameter
}
```

Note that if a function takes multiple type parameters but you want the compiler
to infer some of them, you can use type placeholders:

```
fn foo<T, U>(x: T) {}

fn main() {
    let x: bool = true;
    foo::<f64>(x);    // error, expected 2 parameters, found 1 parameter
    foo::<_, f64>(x); // same as `foo::<bool, f64>(x)`
}
```
"##,

E0106: r##"
This error indicates that a lifetime is missing from a type. If it is an error
inside a function signature, the problem may be with failing to adhere to the
lifetime elision rules (see below).

Here are some simple examples of where you'll run into this error:

```
struct Foo { x: &bool }        // error
struct Foo<'a> { x: &'a bool } // correct

enum Bar { A(u8), B(&bool), }        // error
enum Bar<'a> { A(u8), B(&'a bool), } // correct

type MyStr = &str;        // error
type MyStr<'a> = &'a str; //correct

```

Lifetime elision is a special, limited kind of inference for lifetimes in
function signatures which allows you to leave out lifetimes in certain cases.
For more background on lifetime elision see [the book][book-le].

The lifetime elision rules require that any function signature with an elided
output lifetime must either have

 - exactly one input lifetime
 - or, multiple input lifetimes, but the function must also be a method with a
   `&self` or `&mut self` receiver

In the first case, the output lifetime is inferred to be the same as the unique
input lifetime. In the second case, the lifetime is instead inferred to be the
same as the lifetime on `&self` or `&mut self`.

Here are some examples of elision errors:

```
// error, no input lifetimes
fn foo() -> &str { ... }

// error, `x` and `y` have distinct lifetimes inferred
fn bar(x: &str, y: &str) -> &str { ... }

// error, `y`'s lifetime is inferred to be distinct from `x`'s
fn baz<'a>(x: &'a str, y: &str) -> &str { ... }
```

[book-le]: http://doc.rust-lang.org/nightly/book/lifetimes.html#lifetime-elision
"##,

E0107: r##"
This error means that an incorrect number of lifetime parameters were provided
for a type (like a struct or enum) or trait.

Some basic examples include:

```
struct Foo<'a>(&'a str);
enum Bar { A, B, C }

struct Baz<'a> {
    foo: Foo,     // error: expected 1, found 0
    bar: Bar<'a>, // error: expected 0, found 1
}
```

Here's an example that is currently an error, but may work in a future version
of Rust:

```
struct Foo<'a>(&'a str);

trait Quux { }
impl Quux for Foo { } // error: expected 1, found 0
```

Lifetime elision in implementation headers was part of the lifetime elision
RFC. It is, however, [currently unimplemented][iss15872].

[iss15872]: https://github.com/rust-lang/rust/issues/15872
"##,

E0116: r##"
You can only define an inherent implementation for a type in the same crate
where the type was defined. For example, an `impl` block as below is not allowed
since `Vec` is defined in the standard library:

```
impl Vec<u8> { ... } // error
```

To fix this problem, you can do either of these things:

 - define a trait that has the desired associated functions/types/constants and
   implement the trait for the type in question
 - define a new type wrapping the type and define an implementation on the new
   type

Note that using the `type` keyword does not work here because `type` only
introduces a type alias:

```
type Bytes = Vec<u8>;

impl Bytes { ... } // error, same as above
```
"##,

E0121: r##"
In order to be consistent with Rust's lack of global type inference, type
placeholders are disallowed by design in item signatures.

Examples of this error include:

```
fn foo() -> _ { 5 } // error, explicitly write out the return type instead

static BAR: _ = "test"; // error, explicitly write out the type instead
```
"##,

E0131: r##"
It is not possible to define `main` with type parameters, or even with function
parameters. When `main` is present, it must take no arguments and return `()`.
"##,

E0132: r##"
It is not possible to declare type parameters on a function that has the `start`
attribute. Such a function must have the following type signature:

```
fn(isize, *const *const u8) -> isize
```
"##,

E0166: r##"
This error means that the compiler found a return expression in a function
marked as diverging. A function diverges if it has `!` in the place of the
return type in its signature. For example:

```
fn foo() -> ! { return; } // error
```

For a function that diverges, every control path in the function must never
return, for example with a `loop` that never breaks or a call to another
diverging function (such as `panic!()`).
"##,

E0178: r##"
In types, the `+` type operator has low precedence, so it is often necessary
to use parentheses.

For example:

```
trait Foo {}

struct Bar<'a> {
    w: &'a Foo + Copy,   // error, use &'a (Foo + Copy)
    x: &'a Foo + 'a,     // error, use &'a (Foo + 'a)
    y: &'a mut Foo + 'a, // error, use &'a mut (Foo + 'a)
    z: fn() -> Foo + 'a, // error, use fn() -> (Foo + 'a)
}
```

More details can be found in [RFC 438].

[RFC 438]: https://github.com/rust-lang/rfcs/pull/438
"##,

E0184: r##"
Explicitly implementing both Drop and Copy for a type is currently disallowed.
This feature can make some sense in theory, but the current implementation is
incorrect and can lead to memory unsafety (see [issue #20126][iss20126]), so
it has been disabled for now.

[iss20126]: https://github.com/rust-lang/rust/issues/20126
"##,

E0185: r##"
An associated function for a trait was defined to be static, but an
implementation of the trait declared the same function to be a method (i.e. to
take a `self` parameter).

Here's an example of this error:

```
trait Foo {
    fn foo();
}

struct Bar;

impl Foo for Bar {
    // error, method `foo` has a `&self` declaration in the impl, but not in
    // the trait
    fn foo(&self) {}
}
"##,

E0186: r##"
An associated function for a trait was defined to be a method (i.e. to take a
`self` parameter), but an implementation of the trait declared the same function
to be static.

Here's an example of this error:

```
trait Foo {
    fn foo(&self);
}

struct Bar;

impl Foo for Bar {
    // error, method `foo` has a `&self` declaration in the trait, but not in
    // the impl
    fn foo() {}
}
"##,

E0192: r##"
Negative impls are only allowed for traits with default impls. For more
information see the [opt-in builtin traits RFC](https://github.com/rust-lang/
rfcs/blob/master/text/0019-opt-in-builtin-traits.md).
"##,

E0197: r##"
Inherent implementations (one that do not implement a trait but provide
methods associated with a type) are always safe because they are not
implementing an unsafe trait. Removing the `unsafe` keyword from the inherent
implementation will resolve this error.

```
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

```
struct Foo;

// unsafe is unnecessary
unsafe impl !Clone for Foo { }
// this will compile
impl !Clone for Foo { }
```

"##,

E0199: r##"
Safe traits should not have unsafe implementations, therefore marking an
implementation for a safe trait unsafe will cause a compiler error. Removing the
unsafe marker on the trait noted in the error will resolve this problem.

```
struct Foo;

trait Bar { }

// this won't compile because Bar is safe
unsafe impl Bar for Foo { }
// this will compile
impl Bar for Foo { }
```

"##,

E0200: r##"
Unsafe traits must have unsafe implementations. This error occurs when an
implementation for an unsafe trait isn't marked as unsafe. This may be resolved
by marking the unsafe implementation as unsafe.

```
struct Foo;

unsafe trait Bar { }

// this won't compile because Bar is unsafe and impl isn't unsafe
impl Bar for Foo { }
// this will compile
unsafe impl Bar for Foo { }
```

"##,

E0201: r##"
It is an error to define an associated function more than once.

For example:

```
struct Foo(u8);

impl Foo {
    fn bar(&self) -> bool { self.0 > 5 }

    // error: duplicate associated function
    fn bar() {}
}

trait Baz {
    fn baz(&self) -> bool;
}

impl Baz for Foo {
    fn baz(&self) -> bool { true }

    // error: duplicate method
    fn baz(&self) -> bool { self.0 > 5 }
}
```
"##,

E0202: r##"
Inherent associated types were part of [RFC 195] but are not yet implemented.
See [the tracking issue][iss8995] for the status of this implementation.

[RFC 195]: https://github.com/rust-lang/rfcs/pull/195
[iss8995]: https://github.com/rust-lang/rust/issues/8995
"##,

E0204: r##"
An attempt to implement the `Copy` trait for a struct failed because one of the
fields does not implement `Copy`. To fix this, you must implement `Copy` for the
mentioned field. Note that this may not be possible, as in the example of

```
struct Foo {
    foo : Vec<u32>,
}

impl Copy for Foo { }
```

This fails because `Vec<T>` does not implement `Copy` for any `T`.

Here's another example that will fail:

```
#[derive(Copy)]
struct Foo<'a> {
    ty: &'a mut bool,
}
```

This fails because `&mut T` is not `Copy`, even when `T` is `Copy` (this
differs from the behavior for `&T`, which is always `Copy`).
"##,

E0205: r##"
An attempt to implement the `Copy` trait for an enum failed because one of the
variants does not implement `Copy`. To fix this, you must implement `Copy` for
the mentioned variant. Note that this may not be possible, as in the example of

```
enum Foo {
    Bar(Vec<u32>),
    Baz,
}

impl Copy for Foo { }
```

This fails because `Vec<T>` does not implement `Copy` for any `T`.

Here's another example that will fail:

```
#[derive(Copy)]
enum Foo<'a> {
    Bar(&'a mut bool),
    Baz
}
```

This fails because `&mut T` is not `Copy`, even when `T` is `Copy` (this
differs from the behavior for `&T`, which is always `Copy`).
"##,

E0206: r##"
You can only implement `Copy` for a struct or enum. Both of the following
examples will fail, because neither `i32` (primitive type) nor `&'static Bar`
(reference to `Bar`) is a struct or enum:

```
type Foo = i32;
impl Copy for Foo { } // error

#[derive(Copy, Clone)]
struct Bar;
impl Copy for &'static Bar { } // error
```
"##,

E0243: r##"
This error indicates that not enough type parameters were found in a type or
trait.

For example, the `Foo` struct below is defined to be generic in `T`, but the
type parameter is missing in the definition of `Bar`:

```
struct Foo<T> { x: T }

struct Bar { x: Foo }
```
"##,

E0244: r##"
This error indicates that too many type parameters were found in a type or
trait.

For example, the `Foo` struct below has no type parameters, but is supplied
with two in the definition of `Bar`:

```
struct Foo { x: bool }

struct Bar<S, T> { x: Foo<S, T> }
```
"##,

E0249: r##"
This error indicates a constant expression for the array length was found, but
it was not an integer (signed or unsigned) expression.

Some examples of code that produces this error are:

```
const A: [u32; "hello"] = []; // error
const B: [u32; true] = []; // error
const C: [u32; 0.0] = []; // error
"##,

E0250: r##"
There was an error while evaluating the expression for the length of a fixed-
size array type.

Some examples of this error are:

```
// divide by zero in the length expression
const A: [u32; 1/0] = [];

// Rust currently will not evaluate the function `foo` at compile time
fn foo() -> usize { 12 }
const B: [u32; foo()] = [];

// it is an error to try to add `u8` and `f64`
use std::{f64, u8};
const C: [u32; u8::MAX + f64::EPSILON] = [];
```
"##,

E0318: r##"
Default impls for a trait must be located in the same crate where the trait was
defined. For more information see the [opt-in builtin traits RFC](https://github
.com/rust-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md).
"##,

E0322: r##"
The `Sized` trait is a special trait built-in to the compiler for types with a
constant size known at compile-time. This trait is automatically implemented
for types as needed by the compiler, and it is currently disallowed to
explicitly implement it for a type.
"##,

E0326: r##"
The types of any associated constants in a trait implementation must match the
types in the trait definition. This error indicates that there was a mismatch.

Here's an example of this error:

```
trait Foo {
    const BAR: bool;
}

struct Bar;

impl Foo for Bar {
    const BAR: u32 = 5; // error, expected bool, found u32
}
```
"##,

E0368: r##"
This error indicates that a binary assignment operator like `+=` or `^=` was
applied to the wrong types. For example:

```
let mut x: u16 = 5;
x ^= true; // error, `^=` cannot be applied to types `u16` and `bool`
x += ();   // error, `+=` cannot be applied to types `u16` and `()`
```

Another problem you might be facing is this: suppose you've overloaded the `+`
operator for some type `Foo` by implementing the `std::ops::Add` trait for
`Foo`, but you find that using `+=` does not work, as in this example:

```
use std::ops::Add;

struct Foo(u32);

impl Add for Foo {
    type Output = Foo;

    fn add(self, rhs: Foo) -> Foo {
        Foo(self.0 + rhs.0)
    }
}

fn main() {
    let mut x: Foo = Foo(5);
    x += Foo(7); // error, `+= cannot be applied to types `Foo` and `Foo`
}
```

This is because the binary assignment operators currently do not work off of
traits, so it is not possible to overload them. See [RFC 953] for a proposal
to change this.

[RFC 953]: https://github.com/rust-lang/rfcs/pull/953
"##,

E0371: r##"
When `Trait2` is a subtrait of `Trait1` (for example, when `Trait2` has a
definition like `trait Trait2: Trait1 { ... }`), it is not allowed to implement
`Trait1` for `Trait2`. This is because `Trait2` already implements `Trait1` by
definition, so it is not useful to do this.

Example:

```
trait Foo { fn foo(&self) { } }
trait Bar: Foo { }
trait Baz: Bar { }

impl Bar for Baz { } // error, `Baz` implements `Bar` by definition
impl Foo for Baz { } // error, `Baz` implements `Bar` which implements `Foo`
impl Baz for Baz { } // error, `Baz` (trivially) implements `Baz`
impl Baz for Bar { } // Note: This is OK
```
"##,

E0372: r##"
Trying to implement a trait for a trait object (as in `impl Trait1 for
Trait2 { ... }`) does not work if the trait is not object-safe. Please see the
[RFC 255] for more details on object safety rules.

[RFC 255]: https://github.com/rust-lang/rfcs/pull/255
"##,

E0379: r##"
Trait methods cannot be declared `const` by design. For more information, see
[RFC 911].

[RFC 911]: https://github.com/rust-lang/rfcs/pull/911
"##,

E0380: r##"
Default impls are only allowed for traits with no methods or associated items.
For more information see the [opt-in builtin traits RFC](https://github.com/rust
-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md).
"##

}

register_diagnostics! {
    E0034, // multiple applicable methods in scope
    E0035, // does not take type parameters
    E0036, // incorrect number of type parameters given for this method
    E0044, // foreign items may not have type parameters
    E0045, // variadic function must have C calling convention
    E0068,
    E0071,
    E0074,
    E0075,
    E0076,
    E0077,
    E0085,
    E0086,
    E0088,
    E0090,
    E0091,
    E0092,
    E0093,
    E0094,
    E0101,
    E0102,
    E0103,
    E0104,
    E0117,
    E0118,
    E0119,
    E0120,
    E0122,
    E0123,
    E0124,
    E0127,
    E0128,
    E0129,
    E0130,
    E0141,
    E0159,
    E0163,
    E0164,
    E0167,
    E0168,
    E0172,
    E0173, // manual implementations of unboxed closure traits are experimental
    E0174, // explicit use of unboxed closure methods are experimental
    E0182,
    E0183,
    E0187, // can't infer the kind of the closure
    E0188, // can not cast a immutable reference to a mutable pointer
    E0189, // deprecated: can only cast a boxed pointer to a boxed object
    E0190, // deprecated: can only cast a &-pointer to an &-object
    E0191, // value of the associated type must be specified
    E0193, // cannot bound type where clause bounds may only be attached to types
           // involving type parameters
    E0194,
    E0195, // lifetime parameters or bounds on method do not match the trait declaration
    E0196, // cannot determine a type for this closure
    E0203, // type parameter has more than one relaxed default bound,
           // and only one is supported
    E0207, // type parameter is not constrained by the impl trait, self type, or predicate
    E0208,
    E0209, // builtin traits can only be implemented on structs or enums
    E0210, // type parameter is not constrained by any local type
    E0211,
    E0212, // cannot extract an associated type from a higher-ranked trait bound
    E0213, // associated types are not accepted in this context
    E0214, // parenthesized parameters may only be used with a trait
    E0215, // angle-bracket notation is not stable with `Fn`
    E0216, // parenthetical notation is only stable with `Fn`
    E0217, // ambiguous associated type, defined in multiple supertraits
    E0218, // no associated type defined
    E0219, // associated type defined in higher-ranked supertrait
    E0220, // associated type not found for type parameter
    E0221, // ambiguous associated type in bounds
    E0222, // variadic function must have C calling convention
    E0223, // ambiguous associated type
    E0224, // at least one non-builtin train is required for an object type
    E0225, // only the builtin traits can be used as closure or object bounds
    E0226, // only a single explicit lifetime bound is permitted
    E0227, // ambiguous lifetime bound, explicit lifetime bound required
    E0228, // explicit lifetime bound required
    E0229, // associated type bindings are not allowed here
    E0230, // there is no type parameter on trait
    E0231, // only named substitution parameters are allowed
    E0232, // this attribute must have a value
    E0233,
    E0234,
    E0235, // structure constructor specifies a structure of type but
    E0236, // no lang item for range syntax
    E0237, // no lang item for range syntax
    E0238, // parenthesized parameters may only be used with a trait
    E0239, // `next` method of `Iterator` trait has unexpected type
    E0240,
    E0241,
    E0242, // internal error looking up a definition
    E0245, // not a trait
    E0246, // illegal recursive type
    E0247, // found module name used as a type
    E0248, // found value name used as a type
    E0319, // trait impls for defaulted traits allowed just for structs/enums
    E0320, // recursive overflow during dropck
    E0321, // extended coherence rules for defaulted traits violated
    E0323, // implemented an associated const when another trait item expected
    E0324, // implemented a method when another trait item expected
    E0325, // implemented an associated type when another trait item expected
    E0327, // referred to method instead of constant in match pattern
    E0328, // cannot implement Unsize explicitly
    E0329, // associated const depends on type parameter or Self.
    E0366, // dropck forbid specialization to concrete type or region
    E0367, // dropck forbid specialization to predicate not in struct/enum
    E0369, // binary operation `<op>` cannot be applied to types
    E0374, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures with one field being coerced, none found
    E0375, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures with one field being coerced, but multiple
           // fields need coercions
    E0376, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures
    E0377, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures with the same definition
    E0390, // only a single inherent implementation marked with
           // `#[lang = \"{}\"]` is allowed for the `{}` primitive
    E0391, // unsupported cyclic reference between types/traits detected
    E0392, // parameter `{}` is never used
    E0393  // the type parameter `{}` must be explicitly specified in an object
           // type because its default value `{}` references the type `Self`"
}
