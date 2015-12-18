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
Each field of a struct can only be bound once in a pattern. Erroneous code
example:

```
struct Foo {
    a: u8,
    b: u8,
}

fn main(){
    let x = Foo { a:1, b:2 };

    let Foo { a: x, a: y } = x;
    // error: field `a` bound multiple times in the pattern
}
```

Each occurrence of a field name binds the value of that field, so to fix this
error you will have to remove or alter the duplicate uses of the field name.
Perhaps you misspelled another field name? Example:

```
struct Foo {
    a: u8,
    b: u8,
}

fn main(){
    let x = Foo { a:1, b:2 };

    let Foo { a: x, b: y } = x; // ok!
}
```
"##,

E0026: r##"
This error indicates that a struct pattern attempted to extract a non-existent
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

https://doc.rust-lang.org/reference.html#trait-objects
"##,

E0034: r##"
The compiler doesn't know what method to call because more than one method
has the same prototype. Example:

```
struct Test;

trait Trait1 {
    fn foo();
}

trait Trait2 {
    fn foo();
}

impl Trait1 for Test { fn foo() {} }
impl Trait2 for Test { fn foo() {} }

fn main() {
    Test::foo() // error, which foo() to call?
}
```

To avoid this error, you have to keep only one of them and remove the others.
So let's take our example and fix it:

```
struct Test;

trait Trait1 {
    fn foo();
}

impl Trait1 for Test { fn foo() {} }

fn main() {
    Test::foo() // and now that's good!
}
```

However, a better solution would be using fully explicit naming of type and
trait:

```
struct Test;

trait Trait1 {
    fn foo();
}

trait Trait2 {
    fn foo();
}

impl Trait1 for Test { fn foo() {} }
impl Trait2 for Test { fn foo() {} }

fn main() {
    <Test as Trait1>::foo()
}
```
"##,

E0035: r##"
You tried to give a type parameter where it wasn't needed. Bad example:

```
struct Test;

impl Test {
    fn method(&self) {}
}

fn main() {
    let x = Test;

    x.method::<i32>(); // Error: Test::method doesn't need type parameter!
}
```

To fix this error, just remove the type parameter:

```
struct Test;

impl Test {
    fn method(&self) {}
}

fn main() {
    let x = Test;

    x.method(); // OK, we're good!
}
```
"##,

E0036: r##"
This error occurrs when you pass too many or not enough type parameters to
a method. Example:

```
struct Test;

impl Test {
    fn method<T>(&self, v: &[T]) -> usize {
        v.len()
    }
}

fn main() {
    let x = Test;
    let v = &[0i32];

    x.method::<i32, i32>(v); // error: only one type parameter is expected!
}
```

To fix it, just specify a correct number of type parameters:

```
struct Test;

impl Test {
    fn method<T>(&self, v: &[T]) -> usize {
        v.len()
    }
}

fn main() {
    let x = Test;
    let v = &[0i32];

    x.method::<i32>(v); // OK, we're good!
}
```

Please note on the last example that we could have called `method` like this:

```
x.method(v);
```
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

E0044: r##"
You can't use type parameters on foreign items. Example of erroneous code:

```
extern { fn some_func<T>(x: T); }
```

To fix this, replace the type parameter with the specializations that you
need:

```
extern { fn some_func_i32(x: i32); }
extern { fn some_func_i64(x: i64); }
```
"##,

E0045: r##"
Rust only supports variadic parameters for interoperability with C code in its
FFI. As such, variadic parameters can only be used with functions which are
using the C ABI. Examples of erroneous code:

```
extern "rust-call" { fn foo(x: u8, ...); }
// or
fn foo(x: u8, ...) {}
```

To fix such code, put them in an extern "C" block:

```
extern "C" fn foo (x: u8, ...);
// or:
extern "C" {
    fn foo (x: u8, ...);
}
```
"##,

E0046: r##"
Items are missing in a trait implementation. Erroneous code example:

```
trait Foo {
    fn foo();
}

struct Bar;

impl Foo for Bar {}
// error: not all trait items implemented, missing: `foo`
```

When trying to make some type implement a trait `Foo`, you must, at minimum,
provide implementations for all of `Foo`'s required methods (meaning the
methods that do not have default implementations), as well as any required
trait items like associated types or constants. Example:

```
trait Foo {
    fn foo();
}

struct Bar;

impl Foo for Bar {
    fn foo() {} // ok!
}
```
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
simply calling `printf()` is invalid. But the following uses are allowed:

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
enum variant, one of the fields was specified more than once. Erroneous code
example:

```
struct Foo {
    x: i32
}

fn main() {
    let x = Foo {
                x: 0,
                x: 0, // error: field `x` specified more than once
            };
}
```

Each field should be specified exactly one time. Example:

```
struct Foo {
    x: i32
}

fn main() {
    let x = Foo { x: 0 }; // ok!
}
```
"##,

E0063: r##"
This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was not provided. Erroneous code example:

```
struct Foo {
    x: i32,
    y: i32
}

fn main() {
    let x = Foo { x: 0 }; // error: missing field: `y`
}
```

Each field should be specified exactly once. Example:

```
struct Foo {
    x: i32,
    y: i32
}

fn main() {
    let x = Foo { x: 0, y: 0 }; // ok!
}
```
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
```

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

E0071: r##"
You tried to use structure-literal syntax to create an item that is
not a struct-style structure or enum variant.

Example of erroneous code:

```
enum Foo { FirstValue(i32) };

let u = Foo::FirstValue { value: 0i32 }; // error: Foo::FirstValue
                                         // isn't a structure!
// or even simpler, if the name doesn't refer to a structure at all.
let t = u32 { value: 4 }; // error: `u32` does not name a structure.
```

To fix this, ensure that the name was correctly spelled, and that
the correct form of initializer was used.

For example, the code above can be fixed to:

```
enum Foo {
    FirstValue(i32)
}

fn main() {
    let u = Foo::FirstValue(0i32);

    let t = 4;
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
// error, invalid recursive struct type
struct ListNode {
    head: u8,
    tail: Option<ListNode>,
}
```

This type cannot have a well-defined size, because it needs to be arbitrarily
large (since we would be able to nest `ListNode`s to any depth). Specifically,

```plain
size of `ListNode` = 1 byte for `head`
                   + 1 byte for the discriminant of the `Option`
                   + size of `ListNode`
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

E0074: r##"
When using the `#[simd]` attribute on a tuple struct, the components of the
tuple struct must all be of a concrete, nongeneric type so the compiler can
reason about how to use SIMD with them. This error will occur if the types
are generic.

```
#[simd]
struct Bad<T>(T, T, T); // This will cause an error

#[simd]
struct Good(u32, u32, u32); // This will not
```
"##,

E0075: r##"
The `#[simd]` attribute can only be applied to non empty tuple structs, because
it doesn't make sense to try to use SIMD operations when there are no values to
operate on.

```
#[simd]
struct Bad; // This will cause an error

#[simd]
struct Good(u32); // This will not
```
"##,

E0076: r##"
When using the `#[simd]` attribute to automatically use SIMD operations in tuple
struct, the types in the struct must all be of the same type, or the compiler
will trigger this error.

```
#[simd]
struct Bad(u16, u32, u32); // This will cause an error

#[simd]
struct Good(u32, u32, u32); // This will not
```

"##,

E0077: r##"
When using the `#[simd]` attribute on a tuple struct, the elements in the tuple
must be machine types so SIMD operations can be applied to them.

```
#[simd]
struct Bad(String); // This will cause an error

#[simd]
struct Good(u32, u32, u32); // This will not
```
"##,

E0079: r##"
Enum variants which contain no data can be given a custom integer
representation. This error indicates that the value provided is not an integer
literal and is therefore invalid.

For example, in the following code,

```
enum Foo {
    Q = "32"
}
```

we try to set the representation to a string.

There's no general fix for this; if you can work with an integer then just set
it to one:

```
enum Foo {
    Q = 32
}
```

however if you actually wanted a mapping between variants and non-integer
objects, it may be preferable to use a method with a match instead:

```
enum Foo { Q }
impl Foo {
    fn get_str(&self) -> &'static str {
        match *self {
            Foo::Q => "32",
        }
    }
}
```
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

https://doc.rust-lang.org/reference.html#ffi-attributes
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

The number of supplied parameters must exactly match the number of defined type
parameters.
"##,

E0088: r##"
You gave too many lifetime parameters. Erroneous code example:

```
fn f() {}

fn main() {
    f::<'static>() // error: too many lifetime parameters provided
}
```

Please check you give the right number of lifetime parameters. Example:

```
fn f() {}

fn main() {
    f() // ok!
}
```

It's also important to note that the Rust compiler can generally
determine the lifetime by itself. Example:

```
struct Foo {
    value: String
}

impl Foo {
    // it can be written like this
    fn get_value<'a>(&'a self) -> &'a str { &self.value }
    // but the compiler works fine with this too:
    fn without_lifetime(&self) -> &str { &self.value }
}

fn main() {
    let f = Foo { value: "hello".to_owned() };

    println!("{}", f.get_value());
    println!("{}", f.without_lifetime());
}
```
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

E0091: r##"
You gave an unnecessary type parameter in a type alias. Erroneous code
example:

```
type Foo<T> = u32; // error: type parameter `T` is unused
// or:
type Foo<A,B> = Box<A>; // error: type parameter `B` is unused
```

Please check you didn't write too many type parameters. Example:

```
type Foo = u32; // ok!
type Foo<A> = Box<A>; // ok!
```
"##,

E0092: r##"
You tried to declare an undefined atomic operation function.
Erroneous code example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn atomic_foo(); // error: unrecognized atomic operation
                     //        function
}
```

Please check you didn't make a mistake in the function's name. All intrinsic
functions are defined in librustc_trans/trans/intrinsic.rs and in
libcore/intrinsics.rs in the Rust source code. Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn atomic_fence(); // ok!
}
```
"##,

E0093: r##"
You declared an unknown intrinsic function. Erroneous code example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn foo(); // error: unrecognized intrinsic function: `foo`
}

fn main() {
    unsafe {
        foo();
    }
}
```

Please check you didn't make a mistake in the function's name. All intrinsic
functions are defined in librustc_trans/trans/intrinsic.rs and in
libcore/intrinsics.rs in the Rust source code. Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn atomic_fence(); // ok!
}

fn main() {
    unsafe {
        atomic_fence();
    }
}
```
"##,

E0094: r##"
You gave an invalid number of type parameters to an intrinsic function.
Erroneous code example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T, U>() -> usize; // error: intrinsic has wrong number
                                 //        of type parameters
}
```

Please check that you provided the right number of lifetime parameters
and verify with the function declaration in the Rust source code.
Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T>() -> usize; // ok!
}
```
"##,

E0101: r##"
You hit this error because the compiler lacks the information to
determine a type for this expression. Erroneous code example:

```
fn main() {
    let x = |_| {}; // error: cannot determine a type for this expression
}
```

You have two possibilities to solve this situation:
 * Give an explicit definition of the expression
 * Infer the expression

Examples:

```
fn main() {
    let x = |_ : u32| {}; // ok!
    // or:
    let x = |_| {};
    x(0u32);
}
```
"##,

E0102: r##"
You hit this error because the compiler lacks information to
determine a type for this variable. Erroneous code example:

```
fn demo(devil: fn () -> !) {
    let x: &_ = devil();
    // error: cannot determine a type for this local variable
}

fn oh_no() -> ! { panic!("the devil is in the details") }

fn main() {
    demo(oh_no);
}
```

To solve this situation, constrain the type of the variable.
Examples:

```
fn some_func(x: &u32) {
    // some code
}

fn demo(devil: fn () -> !) {
    let x: &u32 = devil();
    // Here we defined the type at the variable creation

    let x: &_ = devil();
    some_func(x);
    // Here, the type is determined by the function argument type
}

fn oh_no() -> ! { panic!("the devil is in the details") }

fn main() {
    demo(oh_no);
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
type MyStr<'a> = &'a str; // correct
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

[book-le]: https://doc.rust-lang.org/nightly/book/lifetimes.html#lifetime-elision
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

E0117: r##"
This error indicates a violation of one of Rust's orphan rules for trait
implementations. The rule prohibits any implementation of a foreign trait (a
trait defined in another crate) where

 - the type that is implementing the trait is foreign
 - all of the parameters being passed to the trait (if there are any) are also
   foreign.

Here's one example of this error:

```
impl Drop for u32 {}
```

To avoid this kind of error, ensure that at least one local type is referenced
by the `impl`:

```
pub struct Foo; // you define your type in your crate

impl Drop for Foo { // and you can implement the trait on it!
    // code of trait implementation here
}

impl From<Foo> for i32 { // or you use a type from your crate as
                         // a type parameter
    fn from(i: Foo) -> i32 {
        0
    }
}
```

Alternatively, define a trait locally and implement that instead:

```
trait Bar {
    fn get(&self) -> usize;
}

impl Bar for u32 {
    fn get(&self) -> usize { 0 }
}
```

For information on the design of the orphan rules, see [RFC 1023].

[RFC 1023]: https://github.com/rust-lang/rfcs/pull/1023
"##,

E0118: r##"
Rust can't find a base type for an implementation you are providing, or the type
cannot have an implementation. For example, only a named type or a trait can
have an implementation:

```
type NineString = [char, ..9] // This isn't a named type (struct, enum or trait)
impl NineString {
    // Some code here
}
```

In the other, simpler case, Rust just can't find the type you are providing an
impelementation for:

```
impl SomeTypeThatDoesntExist {  }
```
"##,

E0119: r##"
There are conflicting trait implementations for the same type.
Example of erroneous code:

```
trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for T {
    fn get(&self) -> usize { 0 }
}

struct Foo {
    value: usize
}

impl MyTrait for Foo { // error: conflicting implementations of trait
                       //        `MyTrait` for type `Foo`
    fn get(&self) -> usize { self.value }
}
```

When looking for the implementation for the trait, the compiler finds
both the `impl<T> MyTrait for T` where T is all types and the `impl
MyTrait for Foo`. Since a trait cannot be implemented multiple times,
this is an error. So, when you write:

```
impl<T> MyTrait for T {
    fn get(&self) -> usize { 0 }
}
```

This makes the trait implemented on all types in the scope. So if you
try to implement it on another one after that, the implementations will
conflict. Example:

```
trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for T {
    fn get(&self) -> usize { 0 }
}

struct Foo;

fn main() {
    let f = Foo;

    f.get(); // the trait is implemented so we can use it
}
```
"##,

E0120: r##"
An attempt was made to implement Drop on a trait, which is not allowed: only
structs and enums can implement Drop. An example causing this error:

```
trait MyTrait {}

impl Drop for MyTrait {
    fn drop(&mut self) {}
}
```

A workaround for this problem is to wrap the trait up in a struct, and implement
Drop on that. An example is shown below:

```
trait MyTrait {}
struct MyWrapper<T: MyTrait> { foo: T }

impl <T: MyTrait> Drop for MyWrapper<T> {
    fn drop(&mut self) {}
}

```

Alternatively, wrapping trait objects requires something like the following:

```
trait MyTrait {}

//or Box<MyTrait>, if you wanted an owned trait object
struct MyWrapper<'a> { foo: &'a MyTrait }

impl <'a> Drop for MyWrapper<'a> {
    fn drop(&mut self) {}
}
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

E0122: r##"
An attempt was made to add a generic constraint to a type alias. While Rust will
allow this with a warning, it will not currently enforce the constraint.
Consider the example below:

```
trait Foo{}

type MyType<R: Foo> = (R, ());

fn main() {
    let t: MyType<u32>;
}
```

We're able to declare a variable of type `MyType<u32>`, despite the fact that
`u32` does not implement `Foo`. As a result, one should avoid using generic
constraints in concert with type aliases.
"##,

E0124: r##"
You declared two fields of a struct with the same name. Erroneous code
example:

```
struct Foo {
    field1: i32,
    field1: i32 // error: field is already declared
}
```

Please verify that the field names have been correctly spelled. Example:

```
struct Foo {
    field1: i32,
    field2: i32 // ok!
}
```
"##,

E0128: r##"
Type parameter defaults can only use parameters that occur before them.
Erroneous code example:

```
pub struct Foo<T=U, U=()> {
    field1: T,
    filed2: U,
}
// error: type parameters with a default cannot use forward declared
// identifiers
```

Since type parameters are evaluated in-order, you may be able to fix this issue
by doing:

```
pub struct Foo<U=(), T=U> {
    field1: T,
    filed2: U,
}
```

Please also verify that this wasn't because of a name-clash and rename the type
parameter if so.
"##,

E0130: r##"
You declared a pattern as an argument in a foreign function declaration.
Erroneous code example:

```
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
// or
extern {
    fn foo(a: (u32, u32)); // ok!
}
```
"##,

E0131: r##"
It is not possible to define `main` with type parameters, or even with function
parameters. When `main` is present, it must take no arguments and return `()`.
Erroneous code example:

```
fn main<T>() { // error: main function is not allowed to have type parameters
}
```
"##,

E0132: r##"
It is not possible to declare type parameters on a function that has the `start`
attribute. Such a function must have the following type signature:

```
fn(isize, *const *const u8) -> isize;
```
"##,

E0163: r##"
This error means that an attempt was made to match an enum variant as a
struct type when the variant isn't a struct type:

```
enum Foo { B(u32) }

fn bar(foo: Foo) -> u32 {
    match foo {
        Foo::B{i} => i // error 0163
    }
}
```

Try using `()` instead:

```
fn bar(foo: Foo) -> u32 {
    match foo {
        Foo::B(i) => i
    }
}
```
"##,

E0164: r##"

This error means that an attempt was made to match a struct type enum
variant as a non-struct type:

```
enum Foo { B{ i: u32 } }

fn bar(foo: Foo) -> u32 {
    match foo {
        Foo::B(i) => i // error 0164
    }
}
```

Try using `{}` instead:

```
fn bar(foo: Foo) -> u32 {
    match foo {
        Foo::B{i} => i
    }
}
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

E0172: r##"
This error means that an attempt was made to specify the type of a variable with
a combination of a concrete type and a trait. Consider the following example:

```
fn foo(bar: i32+std::fmt::Display) {}
```

The code is trying to specify that we want to receive a signed 32-bit integer
which also implements `Display`. This doesn't make sense: when we pass `i32`, a
concrete type, it implicitly includes all of the traits that it implements.
This includes `Display`, `Debug`, `Clone`, and a host of others.

If `i32` implements the trait we desire, there's no need to specify the trait
separately. If it does not, then we need to `impl` the trait for `i32` before
passing it into `foo`. Either way, a fixed definition for `foo` will look like
the following:

```
fn foo(bar: i32) {}
```

To learn more about traits, take a look at the Book:

https://doc.rust-lang.org/book/traits.html
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
```
"##,

E0191: r##"
Trait objects need to have all associated types specified. Erroneous code
example:

```
trait Trait {
    type Bar;
}

type Foo = Trait; // error: the value of the associated type `Bar` (from
                  //        the trait `Trait`) must be specified
```

Please verify you specified all associated types of the trait and that you
used the right trait. Example:

```
trait Trait {
    type Bar;
}

type Foo = Trait<Bar=i32>; // ok!
```
"##,

E0192: r##"
Negative impls are only allowed for traits with default impls. For more
information see the [opt-in builtin traits RFC](https://github.com/rust-lang/
rfcs/blob/master/text/0019-opt-in-builtin-traits.md).
"##,

E0193: r##"
`where` clauses must use generic type parameters: it does not make sense to use
them otherwise. An example causing this error:

```
trait Foo {
    fn bar(&self);
}

#[derive(Copy,Clone)]
struct Wrapper<T> {
    Wrapped: T
}

impl Foo for Wrapper<u32> where Wrapper<u32>: Clone {
    fn bar(&self) { }
}
```

This use of a `where` clause is strange - a more common usage would look
something like the following:

```
impl <T> Foo for Wrapper<T> where Wrapper<T>: Clone {
    fn bar(&self) { }
}
```

Here, we're saying that the implementation exists on Wrapper only when the
wrapped type `T` implements `Clone`. The `where` clause is important because
some types will not implement `Clone`, and thus will not get this method.

In our erroneous example, however, we're referencing a single concrete type.
Since we know for certain that `Wrapper<u32>` implements `Clone`, there's no
reason to also specify it in a `where` clause.
"##,

E0194: r##"
A type parameter was declared which shadows an existing one. An example of this
error:

```
trait Foo<T> {
    fn do_something(&self) -> T;
    fn do_something_else<T: Clone>(&self, bar: T);
}
```

In this example, the trait `Foo` and the trait method `do_something_else` both
define a type parameter `T`. This is not allowed: if the method wishes to
define a type parameter, it must use a different name for it.
"##,

E0195: r##"
Your method's lifetime parameters do not match the trait declaration.
Erroneous code example:

```
trait Trait {
    fn bar<'a,'b:'a>(x: &'a str, y: &'b str);
}

struct Foo;

impl Trait for Foo {
    fn bar<'a,'b>(x: &'a str, y: &'b str) {
    // error: lifetime parameters or bounds on method `bar`
    // do not match the trait declaration
    }
}
```

The lifetime constraint `'b` for bar() implementation does not match the
trait declaration. Ensure lifetime declarations match exactly in both trait
declaration and implementation. Example:

```
trait Trait {
    fn t<'a,'b:'a>(x: &'a str, y: &'b str);
}

struct Foo;

impl Trait for Foo {
    fn t<'a,'b:'a>(x: &'a str, y: &'b str) { // ok!
    }
}
```
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
It is an error to define two associated items (like methods, associated types,
associated functions, etc.) with the same identifier.

For example:

```
struct Foo(u8);

impl Foo {
    fn bar(&self) -> bool { self.0 > 5 }
    fn bar() {} // error: duplicate associated function
}

trait Baz {
    type Quux;
    fn baz(&self) -> bool;
}

impl Baz for Foo {
    type Quux = u32;

    fn baz(&self) -> bool { true }

    // error: duplicate method
    fn baz(&self) -> bool { self.0 > 5 }

    // error: duplicate associated type
    type Quux = u32;
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

E0207: r##"
You declared an unused type parameter when implementing a trait on an object.
Erroneous code example:

```
trait MyTrait {
    fn get(&self) -> usize;
}

struct Foo;

impl<T> MyTrait for Foo {
    fn get(&self) -> usize {
        0
    }
}
```

Please check your object definition and remove unused type
parameter(s). Example:

```
trait MyTrait {
    fn get(&self) -> usize;
}

struct Foo;

impl MyTrait for Foo {
    fn get(&self) -> usize {
        0
    }
}
```
"##,

E0210: r##"
This error indicates a violation of one of Rust's orphan rules for trait
implementations. The rule concerns the use of type parameters in an
implementation of a foreign trait (a trait defined in another crate), and
states that type parameters must be "covered" by a local type. To understand
what this means, it is perhaps easiest to consider a few examples.

If `ForeignTrait` is a trait defined in some external crate `foo`, then the
following trait `impl` is an error:

```
extern crate foo;
use foo::ForeignTrait;

impl<T> ForeignTrait for T { ... } // error
```

To work around this, it can be covered with a local type, `MyType`:

```
struct MyType<T>(T);
impl<T> ForeignTrait for MyType<T> { ... } // Ok
```

For another example of an error, suppose there's another trait defined in `foo`
named `ForeignTrait2` that takes two type parameters. Then this `impl` results
in the same rule violation:

```
struct MyType2;
impl<T> ForeignTrait2<T, MyType<T>> for MyType2 { ... } // error
```

The reason for this is that there are two appearances of type parameter `T` in
the `impl` header, both as parameters for `ForeignTrait2`. The first appearance
is uncovered, and so runs afoul of the orphan rule.

Consider one more example:

```
impl<T> ForeignTrait2<MyType<T>, T> for MyType2 { ... } // Ok
```

This only differs from the previous `impl` in that the parameters `T` and
`MyType<T>` for `ForeignTrait2` have been swapped. This example does *not*
violate the orphan rule; it is permitted.

To see why that last example was allowed, you need to understand the general
rule. Unfortunately this rule is a bit tricky to state. Consider an `impl`:

```
impl<P1, ..., Pm> ForeignTrait<T1, ..., Tn> for T0 { ... }
```

where `P1, ..., Pm` are the type parameters of the `impl` and `T0, ..., Tn`
are types. One of the types `T0, ..., Tn` must be a local type (this is another
orphan rule, see the explanation for E0117). Let `i` be the smallest integer
such that `Ti` is a local type. Then no type parameter can appear in any of the
`Tj` for `j < i`.

For information on the design of the orphan rules, see [RFC 1023].

[RFC 1023]: https://github.com/rust-lang/rfcs/pull/1023
"##,

E0211: r##"
You used a function or type which doesn't fit the requirements for where it was
used. Erroneous code examples:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T>(); // error: intrinsic has wrong type
}

// or:

fn main() -> i32 { 0 }
// error: main function expects type: `fn() {main}`: expected (), found i32

// or:

let x = 1u8;
match x {
    0u8...3i8 => (),
    // error: mismatched types in range: expected u8, found i8
    _ => ()
}

// or:

use std::rc::Rc;
struct Foo;

impl Foo {
    fn x(self: Rc<Foo>) {}
    // error: mismatched self type: expected `Foo`: expected struct
    //        `Foo`, found struct `alloc::rc::Rc`
}
```

For the first code example, please check the function definition. Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T>() -> usize; // ok!
}
```

The second case example is a bit particular : the main function must always
have this definition:

```
fn main();
```

They never take parameters and never return types.

For the third example, when you match, all patterns must have the same type
as the type you're matching on. Example:

```
let x = 1u8;
match x {
    0u8...3u8 => (), // ok!
    _ => ()
}
```

And finally, for the last example, only `Box<Self>`, `&Self`, `Self`,
or `&mut Self` work as explicit self parameters. Example:

```
struct Foo;

impl Foo {
    fn x(self: Box<Foo>) {} // ok!
}
```
"##,

E0214: r##"
A generic type was described using parentheses rather than angle brackets. For
example:

```
fn main() {
    let v: Vec(&str) = vec!["foo"];
}
```

This is not currently supported: `v` should be defined as `Vec<&str>`.
Parentheses are currently only used with generic types when defining parameters
for `Fn`-family traits.
"##,

E0220: r##"
You used an associated type which isn't defined in the trait.
Erroneous code example:

```
trait Trait {
    type Bar;
}

type Foo = Trait<F=i32>; // error: associated type `F` not found for
                         //        `Trait`
```

Please verify you used the right trait or you didn't misspell the
associated type name. Example:

```
trait Trait {
    type Bar;
}

type Foo = Trait<Bar=i32>; // ok!
```
"##,

E0221: r##"
An attempt was made to retrieve an associated type, but the type was ambiguous.
For example:

```
trait T1 {}
trait T2 {}

trait Foo {
    type A: T1;
}

trait Bar : Foo {
    type A: T2;
    fn do_something() {
        let _: Self::A;
    }
}
```

In this example, `Foo` defines an associated type `A`. `Bar` inherits that type
from `Foo`, and defines another associated type of the same name. As a result,
when we attempt to use `Self::A`, it's ambiguous whether we mean the `A` defined
by `Foo` or the one defined by `Bar`.

There are two options to work around this issue. The first is simply to rename
one of the types. Alternatively, one can specify the intended type using the
following syntax:

```
fn do_something() {
    let _: <Self as Bar>::A;
}
```
"##,

E0223: r##"
An attempt was made to retrieve an associated type, but the type was ambiguous.
For example:

```
trait MyTrait {type X; }

fn main() {
    let foo: MyTrait::X;
}
```

The problem here is that we're attempting to take the type of X from MyTrait.
Unfortunately, the type of X is not defined, because it's only made concrete in
implementations of the trait. A working version of this code might look like:

```
trait MyTrait {type X; }
struct MyStruct;

impl MyTrait for MyStruct {
    type X = u32;
}

fn main() {
    let foo: <MyStruct as MyTrait>::X;
}
```

This syntax specifies that we want the X type from MyTrait, as made concrete in
MyStruct. The reason that we cannot simply use `MyStruct::X` is that MyStruct
might implement two different traits with identically-named associated types.
This syntax allows disambiguation between the two.
"##,

E0225: r##"
You attempted to use multiple types as bounds for a closure or trait object.
Rust does not currently support this. A simple example that causes this error:

```
fn main() {
    let _: Box<std::io::Read+std::io::Write>;
}
```

Builtin traits are an exception to this rule: it's possible to have bounds of
one non-builtin type, plus any number of builtin types. For example, the
following compiles correctly:

```
fn main() {
    let _: Box<std::io::Read+Copy+Sync>;
}
```
"##,

E0232: r##"
The attribute must have a value. Erroneous code example:

```
#[rustc_on_unimplemented] // error: this attribute must have a value
trait Bar {}
```

Please supply the missing value of the attribute. Example:

```
#[rustc_on_unimplemented = "foo"] // ok!
trait Bar {}
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

//NB: not currently reachable
E0247: r##"
This error indicates an attempt to use a module name where a type is expected.
For example:

```
mod MyMod {
    mod MySubMod { }
}

fn do_something(x: MyMod::MySubMod) { }
```

In this example, we're attempting to take a parameter of type `MyMod::MySubMod`
in the do_something function. This is not legal: `MyMod::MySubMod` is a module
name, not a type.
"##,

E0248: r##"
This error indicates an attempt to use a value where a type is expected. For
example:

```
enum Foo {
    Bar(u32)
}

fn do_something(x: Foo::Bar) { }
```

In this example, we're attempting to take a type of `Foo::Bar` in the
do_something function. This is not legal: `Foo::Bar` is a value of type `Foo`,
not a distinct static type. Likewise, it's not legal to attempt to
`impl Foo::Bar`: instead, you must `impl Foo` and then pattern match to specify
behavior for specific enum variants.
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

E0321: r##"
A cross-crate opt-out trait was implemented on something which wasn't a struct
or enum type. Erroneous code example:

```
#![feature(optin_builtin_traits)]

struct Foo;

impl !Sync for Foo {}

unsafe impl Send for &'static Foo {
// error: cross-crate traits with a default impl, like `core::marker::Send`,
//        can only be implemented for a struct/enum type, not
//        `&'static Foo`
```

Only structs and enums are permitted to impl Send, Sync, and other opt-out
trait, and the struct or enum must be local to the current crate. So, for
example, `unsafe impl Send for Rc<Foo>` is not allowed.
"##,

E0322: r##"
The `Sized` trait is a special trait built-in to the compiler for types with a
constant size known at compile-time. This trait is automatically implemented
for types as needed by the compiler, and it is currently disallowed to
explicitly implement it for a type.
"##,

E0323: r##"
An associated const was implemented when another trait item was expected.
Erroneous code example:

```
trait Foo {
    type N;
}

struct Bar;

impl Foo for Bar {
    const N : u32 = 0;
    // error: item `N` is an associated const, which doesn't match its
    //        trait `<Bar as Foo>`
}
```

Please verify that the associated const wasn't misspelled and the correct trait
was implemented. Example:

```
struct Bar;

trait Foo {
    type N;
}

impl Foo for Bar {
    type N = u32; // ok!
}

// or:
trait Foo {
    const N : u32;
}

impl Foo for Bar {
    const N : u32 = 0; // ok!
}
```
"##,

E0324: r##"
A method was implemented when another trait item was expected. Erroneous
code example:

```
struct Bar;

trait Foo {
    const N : u32;

    fn M();
}

impl Foo for Bar {
    fn N() {}
    // error: item `N` is an associated method, which doesn't match its
    //        trait `<Bar as Foo>`
}
```

To fix this error, please verify that the method name wasn't misspelled and
verify that you are indeed implementing the correct trait items. Example:

```
struct Bar;

trait Foo {
    const N : u32;

    fn M();
}

impl Foo for Bar {
    const N : u32 = 0;

    fn M() {} // ok!
}
```
"##,

E0325: r##"
An associated type was implemented when another trait item was expected.
Erroneous code example:

```
struct Bar;

trait Foo {
    const N : u32;
}

impl Foo for Bar {
    type N = u32;
    // error: item `N` is an associated type, which doesn't match its
    //        trait `<Bar as Foo>`
}
```

Please verify that the associated type name wasn't misspelled and your
implementation corresponds to the trait definition. Example:

```
struct Bar;

trait Foo {
    type N;
}

impl Foo for Bar {
    type N = u32; // ok!
}

//or:
trait Foo {
    const N : u32;
}

impl Foo for Bar {
    const N : u32 = 0; // ok!
}
```
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

E0327: r##"
You cannot use associated items other than constant items as patterns. This
includes method items. Example of erroneous code:

```
enum B {}

impl B {
    fn bb() -> i32 { 0 }
}

fn main() {
    match 0 {
        B::bb => {} // error: associated items in match patterns must
                    // be constants
    }
}
```

Please check that you're not using a method as a pattern. Example:

```
enum B {
    ba,
    bb
}

fn main() {
    match B::ba {
        B::bb => {} // ok!
        _ => {}
    }
}
```
"##,

E0329: r##"
An attempt was made to access an associated constant through either a generic
type parameter or `Self`. This is not supported yet. An example causing this
error is shown below:

```
trait Foo {
    const BAR: f64;
}

struct MyStruct;

impl Foo for MyStruct {
    const BAR: f64 = 0f64;
}

fn get_bar_bad<F: Foo>(t: F) -> f64 {
    F::BAR
}
```

Currently, the value of `BAR` for a particular type can only be accessed through
a concrete type, as shown below:

```
fn get_bar_good() -> f64 {
    <MyStruct as Foo>::BAR
}
```
"##,

E0366: r##"
An attempt was made to implement `Drop` on a concrete specialization of a
generic type. An example is shown below:

```
struct Foo<T> {
    t: T
}

impl Drop for Foo<u32> {
    fn drop(&mut self) {}
}
```

This code is not legal: it is not possible to specialize `Drop` to a subset of
implementations of a generic type. One workaround for this is to wrap the
generic type, as shown below:

```
struct Foo<T> {
    t: T
}

struct Bar {
    t: Foo<u32>
}

impl Drop for Bar {
    fn drop(&mut self) {}
}
```
"##,

E0367: r##"
An attempt was made to implement `Drop` on a specialization of a generic type.
An example is shown below:

```
trait Foo{}

struct MyStruct<T> {
    t: T
}

impl<T: Foo> Drop for MyStruct<T> {
    fn drop(&mut self) {}
}
```

This code is not legal: it is not possible to specialize `Drop` to a subset of
implementations of a generic type. In order for this code to work, `MyStruct`
must also require that `T` implements `Foo`. Alternatively, another option is
to wrap the generic type in another that specializes appropriately:

```
trait Foo{}

struct MyStruct<T> {
    t: T
}

struct MyStructWrapper<T: Foo> {
    t: MyStruct<T>
}

impl <T: Foo> Drop for MyStructWrapper<T> {
    fn drop(&mut self) {}
}
```
"##,

E0368: r##"
This error indicates that a binary assignment operator like `+=` or `^=` was
applied to a type that doesn't support it. For example:

```
let mut x = 12f32; // error: binary operation `<<` cannot be applied to
               //        type `f32`

x <<= 2;
```

To fix this error, please check that this type implements this binary
operation. Example:

```
let x = 12u32; // the `u32` type does implement the `ShlAssign` trait

x <<= 2; // ok!
```

It is also possible to overload most operators for your own type by
implementing the `[OP]Assign` traits from `std::ops`.

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
    x += Foo(7); // error, `+= cannot be applied to the type `Foo`
}
```

This is because `AddAssign` is not automatically implemented, so you need to
manually implement it for your type.
"##,

E0369: r##"
A binary operation was attempted on a type which doesn't support it.
Erroneous code example:

```
let x = 12f32; // error: binary operation `<<` cannot be applied to
               //        type `f32`

x << 2;
```

To fix this error, please check that this type implements this binary
operation. Example:

```
let x = 12u32; // the `u32` type does implement it:
               // https://doc.rust-lang.org/stable/std/ops/trait.Shl.html

x << 2; // ok!
```

It is also possible to overload most operators for your own type by
implementing traits from `std::ops`.
"##,

E0370: r##"
The maximum value of an enum was reached, so it cannot be automatically
set in the next enum value. Erroneous code example:

```
enum Foo {
    X = 0x7fffffffffffffff,
    Y // error: enum discriminant overflowed on value after
      //        9223372036854775807: i64; set explicitly via
      //        Y = -9223372036854775808 if that is desired outcome
}
```

To fix this, please set manually the next enum value or put the enum variant
with the maximum value at the end of the enum. Examples:

```
enum Foo {
    X = 0x7fffffffffffffff,
    Y = 0, // ok!
}

// or:
enum Foo {
    Y = 0, // ok!
    X = 0x7fffffffffffffff,
}
```
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

E0379: r##"
Trait methods cannot be declared `const` by design. For more information, see
[RFC 911].

[RFC 911]: https://github.com/rust-lang/rfcs/pull/911
"##,

E0380: r##"
Default impls are only allowed for traits with no methods or associated items.
For more information see the [opt-in builtin traits RFC](https://github.com/rust
-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md).
"##,

E0390: r##"
You tried to implement methods for a primitive type. Erroneous code example:

```
struct Foo {
    x: i32
}

impl *mut Foo {}
// error: only a single inherent implementation marked with
//        `#[lang = "mut_ptr"]` is allowed for the `*mut T` primitive
```

This isn't allowed, but using a trait to implement a method is a good solution.
Example:

```
struct Foo {
    x: i32
}

trait Bar {
    fn bar();
}

impl Bar for *mut Foo {
    fn bar() {} // ok!
}
```
"##,

E0391: r##"
This error indicates that some types or traits depend on each other
and therefore cannot be constructed.

The following example contains a circular dependency between two traits:

```
trait FirstTrait : SecondTrait {

}

trait SecondTrait : FirstTrait {

}
```
"##,

E0392: r##"
This error indicates that a type or lifetime parameter has been declared
but not actually used.  Here is an example that demonstrates the error:

```
enum Foo<T> {
    Bar
}
```

If the type parameter was included by mistake, this error can be fixed
by simply removing the type parameter, as shown below:

```
enum Foo {
    Bar
}
```

Alternatively, if the type parameter was intentionally inserted, it must be
used. A simple fix is shown below:

```
enum Foo<T> {
    Bar(T)
}
```

This error may also commonly be found when working with unsafe code. For
example, when using raw pointers one may wish to specify the lifetime for
which the pointed-at data is valid. An initial attempt (below) causes this
error:

```
struct Foo<'a, T> {
    x: *const T
}
```

We want to express the constraint that Foo should not outlive `'a`, because
the data pointed to by `T` is only valid for that lifetime. The problem is
that there are no actual uses of `'a`. It's possible to work around this
by adding a PhantomData type to the struct, using it to tell the compiler
to act as if the struct contained a borrowed reference `&'a T`:

```
use std::marker::PhantomData;

struct Foo<'a, T: 'a> {
    x: *const T,
    phantom: PhantomData<&'a T>
}
```

PhantomData can also be used to express information about unused type
parameters. You can read more about it in the API documentation:

https://doc.rust-lang.org/std/marker/struct.PhantomData.html
"##,

E0439: r##"
The length of the platform-intrinsic function `simd_shuffle`
wasn't specified. Erroneous code example:

```
extern "platform-intrinsic" {
    fn simd_shuffle<A,B>(a: A, b: A, c: [u32; 8]) -> B;
    // error: invalid `simd_shuffle`, needs length: `simd_shuffle`
}
```

The `simd_shuffle` function needs the length of the array passed as
last parameter in its name. Example:

```
extern "platform-intrinsic" {
    fn simd_shuffle8<A,B>(a: A, b: A, c: [u32; 8]) -> B;
}
```
"##,

E0440: r##"
A platform-specific intrinsic function has the wrong number of type
parameters. Erroneous code example:

```
#[repr(simd)]
struct f64x2(f64, f64);

extern "platform-intrinsic" {
    fn x86_mm_movemask_pd<T>(x: f64x2) -> i32;
    // error: platform-specific intrinsic has wrong number of type
    //        parameters
}
```

Please refer to the function declaration to see if it corresponds
with yours. Example:

```
#[repr(simd)]
struct f64x2(f64, f64);

extern "platform-intrinsic" {
    fn x86_mm_movemask_pd(x: f64x2) -> i32;
}
```
"##,

E0441: r##"
An unknown platform-specific intrinsic function was used. Erroneous
code example:

```
#[repr(simd)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

extern "platform-intrinsic" {
    fn x86_mm_adds_ep16(x: i16x8, y: i16x8) -> i16x8;
    // error: unrecognized platform-specific intrinsic function
}
```

Please verify that the function name wasn't misspelled, and ensure
that it is declared in the rust source code (in the file
src/librustc_platform_intrinsics/x86.rs). Example:

```
#[repr(simd)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

extern "platform-intrinsic" {
    fn x86_mm_adds_epi16(x: i16x8, y: i16x8) -> i16x8; // ok!
}
```
"##,

E0442: r##"
Intrinsic argument(s) and/or return value have the wrong type.
Erroneous code example:

```
#[repr(simd)]
struct i8x16(i8, i8, i8, i8, i8, i8, i8, i8,
             i8, i8, i8, i8, i8, i8, i8, i8);
#[repr(simd)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
struct i64x2(i64, i64);

extern "platform-intrinsic" {
    fn x86_mm_adds_epi16(x: i8x16, y: i32x4) -> i64x2;
    // error: intrinsic arguments/return value have wrong type
}
```

To fix this error, please refer to the function declaration to give
it the awaited types. Example:

```
#[repr(simd)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

extern "platform-intrinsic" {
    fn x86_mm_adds_epi16(x: i16x8, y: i16x8) -> i16x8; // ok!
}
```
"##,

E0443: r##"
Intrinsic argument(s) and/or return value have the wrong type.
Erroneous code example:

```
#[repr(simd)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);
#[repr(simd)]
struct i64x8(i64, i64, i64, i64, i64, i64, i64, i64);

extern "platform-intrinsic" {
    fn x86_mm_adds_epi16(x: i16x8, y: i16x8) -> i64x8;
    // error: intrinsic argument/return value has wrong type
}
```

To fix this error, please refer to the function declaration to give
it the awaited types. Example:

```
#[repr(simd)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

extern "platform-intrinsic" {
    fn x86_mm_adds_epi16(x: i16x8, y: i16x8) -> i16x8; // ok!
}
```
"##,

E0444: r##"
A platform-specific intrinsic function has wrong number of arguments.
Erroneous code example:

```
#[repr(simd)]
struct f64x2(f64, f64);

extern "platform-intrinsic" {
    fn x86_mm_movemask_pd(x: f64x2, y: f64x2, z: f64x2) -> i32;
    // error: platform-specific intrinsic has invalid number of arguments
}
```

Please refer to the function declaration to see if it corresponds
with yours. Example:

```
#[repr(simd)]
struct f64x2(f64, f64);

extern "platform-intrinsic" {
    fn x86_mm_movemask_pd(x: f64x2) -> i32; // ok!
}
```
"##,

E0516: r##"
The `typeof` keyword is currently reserved but unimplemented.
Erroneous code example:

```
fn main() {
    let x: typeof(92) = 92;
}
```

Try using type inference instead. Example:

```
fn main() {
    let x = 92;
}
```
"##,

}

register_diagnostics! {
//  E0068,
//  E0085,
//  E0086,
    E0090,
    E0103, // @GuillaumeGomez: I was unable to get this error, try your best!
    E0104,
//  E0123,
//  E0127,
//  E0129,
//  E0141,
//  E0159, // use of trait `{}` as struct constructor
    E0167,
//  E0168,
//  E0173, // manual implementations of unboxed closure traits are experimental
    E0174, // explicit use of unboxed closure methods are experimental
    E0182,
    E0183,
//  E0187, // can't infer the kind of the closure
//  E0188, // can not cast an immutable reference to a mutable pointer
//  E0189, // deprecated: can only cast a boxed pointer to a boxed object
//  E0190, // deprecated: can only cast a &-pointer to an &-object
    E0196, // cannot determine a type for this closure
    E0203, // type parameter has more than one relaxed default bound,
           // and only one is supported
    E0208,
//  E0209, // builtin traits can only be implemented on structs or enums
    E0212, // cannot extract an associated type from a higher-ranked trait bound
//  E0213, // associated types are not accepted in this context
//  E0215, // angle-bracket notation is not stable with `Fn`
//  E0216, // parenthetical notation is only stable with `Fn`
//  E0217, // ambiguous associated type, defined in multiple supertraits
//  E0218, // no associated type defined
//  E0219, // associated type defined in higher-ranked supertrait
//  E0222, // Error code E0045 (variadic function must have C calling
           // convention) duplicate
    E0224, // at least one non-builtin train is required for an object type
    E0226, // only a single explicit lifetime bound is permitted
    E0227, // ambiguous lifetime bound, explicit lifetime bound required
    E0228, // explicit lifetime bound required
    E0230, // there is no type parameter on trait
    E0231, // only named substitution parameters are allowed
//  E0233,
//  E0234,
//  E0235, // structure constructor specifies a structure of type but
    E0236, // no lang item for range syntax
    E0237, // no lang item for range syntax
    E0238, // parenthesized parameters may only be used with a trait
//  E0239, // `next` method of `Iterator` trait has unexpected type
//  E0240,
//  E0241,
    E0242, // internal error looking up a definition
    E0245, // not a trait
//  E0246, // invalid recursive type
//  E0319, // trait impls for defaulted traits allowed just for structs/enums
    E0320, // recursive overflow during dropck
    E0328, // cannot implement Unsize explicitly
//  E0372, // coherence not object safe
    E0374, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures with one field being coerced, none found
    E0375, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures with one field being coerced, but multiple
           // fields need coercions
    E0376, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures
    E0377, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures with the same definition
    E0393, // the type parameter `{}` must be explicitly specified in an object
           // type because its default value `{}` references the type `Self`"
    E0399, // trait items need to be implemented because the associated
           // type `{}` was overridden
    E0436, // functional record update requires a struct
    E0513  // no type for local variable ..
}
