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
    Apple(String, String),
    Pear(u32),
}
```

Here the `Apple` variant has two fields, and should be matched against like so:

```
enum Fruit {
    Apple(String, String),
    Pear(u32),
}

let x = Fruit::Apple(String::new(), String::new());

// Correct.
match x {
    Fruit::Apple(a, b) => {},
    _ => {}
}
```

Matching with the wrong number of fields has no sensible interpretation:

```compile_fail,E0023
enum Fruit {
    Apple(String, String),
    Pear(u32),
}

let x = Fruit::Apple(String::new(), String::new());

// Incorrect.
match x {
    Fruit::Apple(a) => {},
    Fruit::Apple(a, b, c) => {},
}
```

Check how many fields the enum was declared with and ensure that your pattern
uses the same number.
"##,

E0025: r##"
Each field of a struct can only be bound once in a pattern. Erroneous code
example:

```compile_fail,E0025
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
    Thing { x: xfield, y: yfield } => {}
}
```

If you are using shorthand field patterns but want to refer to the struct field
by a different name, you should rename it explicitly.

Change this:

```compile_fail,E0026
struct Thing {
    x: u32,
    y: u32
}

let thing = Thing { x: 0, y: 0 };

match thing {
    Thing { x, z } => {}
}
```

To this:

```
struct Thing {
    x: u32,
    y: u32
}

let thing = Thing { x: 0, y: 0 };

match thing {
    Thing { x, y: z } => {}
}
```
"##,

E0027: r##"
This error indicates that a pattern for a struct fails to specify a sub-pattern
for every one of the struct's fields. Ensure that each field from the struct's
definition is mentioned in the pattern, or use `..` to ignore unwanted fields.

For example:

```compile_fail,E0027
struct Dog {
    name: String,
    age: u32,
}

let d = Dog { name: "Rusty".to_string(), age: 8 };

// This is incorrect.
match d {
    Dog { age: x } => {}
}
```

This is correct (explicit):

```
struct Dog {
    name: String,
    age: u32,
}

let d = Dog { name: "Rusty".to_string(), age: 8 };

match d {
    Dog { name: ref n, age: x } => {}
}

// This is also correct (ignore unused fields).
match d {
    Dog { age: x, .. } => {}
}
```
"##,

E0029: r##"
In a match expression, only numbers and characters can be matched against a
range. This is because the compiler checks that the range is non-empty at
compile-time, and is unable to evaluate arbitrary comparison functions. If you
want to capture values of an orderable type between two end-points, you can use
a guard.

```compile_fail,E0029
let string = "salutations !";

// The ordering relation for strings can't be evaluated at compile time,
// so this doesn't work:
match string {
    "hello" ... "world" => {}
    _ => {}
}

// This is a more general version, using a guard:
match string {
    s if s >= "hello" && s <= "world" => {}
    _ => {}
}
```
"##,

E0033: r##"
This error indicates that a pointer to a trait type cannot be implicitly
dereferenced by a pattern. Every trait defines a type, but because the
size of trait implementors isn't fixed, this type has no compile-time size.
Therefore, all accesses to trait types must be through pointers. If you
encounter this error you should try to avoid dereferencing the pointer.

```ignore
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
has the same prototype. Erroneous code example:

```compile_fail,E0034
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

One last example:

```
trait F {
    fn m(&self);
}

trait G {
    fn m(&self);
}

struct X;

impl F for X { fn m(&self) { println!("I am F"); } }
impl G for X { fn m(&self) { println!("I am G"); } }

fn main() {
    let f = X;

    F::m(&f); // it displays "I am F"
    G::m(&f); // it displays "I am G"
}
```
"##,

E0035: r##"
You tried to give a type parameter where it wasn't needed. Erroneous code
example:

```compile_fail,E0035
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
a method. Erroneous code example:

```compile_fail,E0036
struct Test;

impl Test {
    fn method<T>(&self, v: &[T]) -> usize {
        v.len()
    }
}

fn main() {
    let x = Test;
    let v = &[0];

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
    let v = &[0];

    x.method::<i32>(v); // OK, we're good!
}
```

Please note on the last example that we could have called `method` like this:

```ignore
x.method(v);
```
"##,

E0040: r##"
It is not allowed to manually call destructors in Rust. It is also not
necessary to do this since `drop` is called automatically whenever a value goes
out of scope.

Here's an example of this error:

```compile_fail,E0040
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

```compile_fail,E0044
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

```compile_fail
#![feature(unboxed_closures)]

extern "rust-call" { fn foo(x: u8, ...); }

// or

fn foo(x: u8, ...) {}
```

To fix such code, put them in an extern "C" block:

```
extern "C" {
    fn foo (x: u8, ...);
}
```
"##,

E0046: r##"
Items are missing in a trait implementation. Erroneous code example:

```compile_fail,E0046
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

```compile_fail,E0049
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

```compile_fail,E0050
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

```compile_fail,E0053
trait Foo {
    fn foo(x: u16);
    fn bar(&self);
}

struct Bar;

impl Foo for Bar {
    // error, expected u16, found i16
    fn foo(x: i16) { }

    // error, types differ in mutability
    fn bar(&mut self) { }
}
```
"##,

E0054: r##"
It is not allowed to cast to a bool. If you are trying to cast a numeric type
to a bool, you can compare it with zero instead:

```compile_fail,E0054
let x = 5;

// Not allowed, won't compile
let x_is_nonzero = x as bool;
```

```
let x = 5;

// Ok
let x_is_nonzero = x != 0;
```
"##,

E0055: r##"
During a method call, a value is automatically dereferenced as many times as
needed to make the value's type match the method's receiver. The catch is that
the compiler will only attempt to dereference a number of times up to the
recursion limit (which can be set via the `recursion_limit` attribute).

For a somewhat artificial example:

```compile_fail,E0055
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

```compile_fail,E0057
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

```compile_fail,E0059
#![feature(unboxed_closures)]

fn foo<F: Fn<i32>>(f: F) -> F::Output { f(3) }
```

It can be fixed by adjusting the trait bound like this:

```
#![feature(unboxed_closures)]

fn foo<F: Fn<(i32,)>>(f: F) -> F::Output { f(3) }
```

Note that `(T,)` always denotes the type of a 1-tuple containing an element of
type `T`. The comma is necessary for syntactic disambiguation.
"##,

E0060: r##"
External C functions are allowed to be variadic. However, a variadic function
takes a minimum number of arguments. For example, consider C's variadic `printf`
function:

```ignore
extern crate libc;
use libc::{ c_char, c_int };

extern "C" {
    fn printf(_: *const c_char, ...) -> c_int;
}
```

Using this declaration, it must be called with at least one argument, so
simply calling `printf()` is invalid. But the following uses are allowed:

```ignore
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

For example, a function like:

```
fn f(a: u16, b: &str) {}
```

Must always be called with exactly two arguments, e.g. `f(2, "test")`.

Note that Rust does not have a notion of optional function arguments or
variadic functions (except for its C-FFI).
"##,

E0062: r##"
This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was specified more than once. Erroneous code
example:

```compile_fail,E0062
struct Foo {
    x: i32,
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
    x: i32,
}

fn main() {
    let x = Foo { x: 0 }; // ok!
}
```
"##,

E0063: r##"
This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was not provided. Erroneous code example:

```compile_fail,E0063
struct Foo {
    x: i32,
    y: i32,
}

fn main() {
    let x = Foo { x: 0 }; // error: missing field: `y`
}
```

Each field should be specified exactly once. Example:

```
struct Foo {
    x: i32,
    y: i32,
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

Let's start with some erroneous code examples:

```compile_fail,E0067
use std::collections::LinkedList;

// Bad: assignment to non-lvalue expression
LinkedList::new() += 1;

// ...

fn some_func(i: &mut i32) {
    i += 12; // Error : '+=' operation cannot be applied on a reference !
}
```

And now some working examples:

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

```compile_fail,E0069
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
https://doc.rust-lang.org/reference.html#lvalues-rvalues-and-temporaries

Now, we can go further. Here are some erroneous code examples:

```compile_fail,E0070
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

And now let's give working examples:

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
not a structure or enum variant.

Example of erroneous code:

```compile_fail,E0071
type U32 = u32;
let t = U32 { value: 4 }; // error: expected struct, variant or union type,
                          // found builtin type `u32`
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

E0073: r##"
You cannot define a struct (or enum) `Foo` that requires an instance of `Foo`
in order to make a new `Foo` value. This is because there would be no way a
first instance of `Foo` could be made to initialize another instance!

Here's an example of a struct that has this problem:

```ignore
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

This will cause an error:

```ignore
#![feature(repr_simd)]

#[repr(simd)]
struct Bad<T>(T, T, T);
```

This will not:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32, u32, u32);
```
"##,

E0075: r##"
The `#[simd]` attribute can only be applied to non empty tuple structs, because
it doesn't make sense to try to use SIMD operations when there are no values to
operate on.

This will cause an error:

```compile_fail,E0075
#![feature(repr_simd)]

#[repr(simd)]
struct Bad;
```

This will not:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32);
```
"##,

E0076: r##"
When using the `#[simd]` attribute to automatically use SIMD operations in tuple
struct, the types in the struct must all be of the same type, or the compiler
will trigger this error.

This will cause an error:

```compile_fail,E0076
#![feature(repr_simd)]

#[repr(simd)]
struct Bad(u16, u32, u32);
```

This will not:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32, u32, u32);
```
"##,

E0077: r##"
When using the `#[simd]` attribute on a tuple struct, the elements in the tuple
must be machine types so SIMD operations can be applied to them.

This will cause an error:

```compile_fail,E0077
#![feature(repr_simd)]

#[repr(simd)]
struct Bad(String);
```

This will not:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32, u32, u32);
```
"##,

E0079: r##"
Enum variants which contain no data can be given a custom integer
representation. This error indicates that the value provided is not an integer
literal and is therefore invalid.

For example, in the following code:

```compile_fail,E0079
enum Foo {
    Q = "32",
}
```

We try to set the representation to a string.

There's no general fix for this; if you can work with an integer then just set
it to one:

```
enum Foo {
    Q = 32,
}
```

However if you actually wanted a mapping between variants and non-integer
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

E0081: r##"
Enum discriminants are used to differentiate enum variants stored in memory.
This error indicates that the same value was used for two or more variants,
making them impossible to tell apart.

```compile_fail,E0081
// Bad.
enum Enum {
    P = 3,
    X = 3,
    Y = 5,
}
```

```
// Good.
enum Enum {
    P,
    X = 3,
    Y = 5,
}
```

Note that variants without a manually specified discriminant are numbered from
top to bottom starting from 0, so clashes can occur with seemingly unrelated
variants.

```compile_fail,E0081
enum Bad {
    X,
    Y = 0
}
```

Here `X` will have already been specified the discriminant 0 by the time `Y` is
encountered, so a conflict occurs.
"##,

E0082: r##"
When you specify enum discriminants with `=`, the compiler expects `isize`
values by default. Or you can add the `repr` attibute to the enum declaration
for an explicit choice of the discriminant type. In either cases, the
discriminant values must fall within a valid range for the expected type;
otherwise this error is raised. For example:

```ignore
#[repr(u8)]
enum Thing {
    A = 1024,
    B = 5,
}
```

Here, 1024 lies outside the valid range for `u8`, so the discriminant for `A` is
invalid. Here is another, more subtle example which depends on target word size:

```ignore
enum DependsOnPointerSize {
    A = 1 << 32,
}
```

Here, `1 << 32` is interpreted as an `isize` value. So it is invalid for 32 bit
target (`target_pointer_width = "32"`) but valid for 64 bit target.

You may want to change representation types to fix this, or else change invalid
discriminant values so that they fit within the existing type.
"##,

E0084: r##"
An unsupported representation was attempted on a zero-variant enum.

Erroneous code example:

```compile_fail,E0084
#[repr(i32)]
enum NightsWatch {} // error: unsupported representation for zero-variant enum
```

It is impossible to define an integer type to be used to represent zero-variant
enum values because there are no zero-variant enum values. There is no way to
construct an instance of the following type using only safe code. So you have
two solutions. Either you add variants in your enum:

```
#[repr(i32)]
enum NightsWatch {
    JonSnow,
    Commander,
}
```

or you remove the integer represention of your enum:

```
enum NightsWatch {}
```
"##,

E0087: r##"
Too many type parameters were supplied for a function. For example:

```compile_fail,E0087
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

```compile_fail,E0088
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

```compile_fail,E0089
fn foo<T, U>() {}

fn main() {
    foo::<f64>(); // error, expected 2 parameters, found 1 parameter
}
```

Note that if a function takes multiple type parameters but you want the compiler
to infer some of them, you can use type placeholders:

```compile_fail,E0089
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

```compile_fail,E0091
type Foo<T> = u32; // error: type parameter `T` is unused
// or:
type Foo<A,B> = Box<A>; // error: type parameter `B` is unused
```

Please check you didn't write too many type parameters. Example:

```
type Foo = u32; // ok!
type Foo2<A> = Box<A>; // ok!
```
"##,

E0092: r##"
You tried to declare an undefined atomic operation function.
Erroneous code example:

```compile_fail,E0092
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

```compile_fail,E0093
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

```compile_fail,E0094
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T, U>() -> usize; // error: intrinsic has wrong number
                                 //        of type parameters
}
```

Please check that you provided the right number of type parameters
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

```compile_fail,E0101
let x = |_| {}; // error: cannot determine a type for this expression
```

You have two possibilities to solve this situation:

* Give an explicit definition of the expression
* Infer the expression

Examples:

```
let x = |_ : u32| {}; // ok!
// or:
let x = |_| {};
x(0u32);
```
"##,

E0102: r##"
You hit this error because the compiler lacks the information to
determine the type of this variable. Erroneous code example:

```compile_fail,E0102
// could be an array of anything
let x = []; // error: cannot determine a type for this local variable
```

To solve this situation, constrain the type of the variable.
Examples:

```
#![allow(unused_variables)]

fn main() {
    let x: [u8; 0] = [];
}
```
"##,

E0106: r##"
This error indicates that a lifetime is missing from a type. If it is an error
inside a function signature, the problem may be with failing to adhere to the
lifetime elision rules (see below).

Here are some simple examples of where you'll run into this error:

```compile_fail,E0106
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

```compile_fail,E0106
// error, no input lifetimes
fn foo() -> &str { }

// error, `x` and `y` have distinct lifetimes inferred
fn bar(x: &str, y: &str) -> &str { }

// error, `y`'s lifetime is inferred to be distinct from `x`'s
fn baz<'a>(x: &'a str, y: &str) -> &str { }
```

[book-le]: https://doc.rust-lang.org/nightly/book/lifetimes.html#lifetime-elision
"##,

E0107: r##"
This error means that an incorrect number of lifetime parameters were provided
for a type (like a struct or enum) or trait.

Some basic examples include:

```compile_fail,E0107
struct Foo<'a>(&'a str);
enum Bar { A, B, C }

struct Baz<'a> {
    foo: Foo,     // error: expected 1, found 0
    bar: Bar<'a>, // error: expected 0, found 1
}
```

Here's an example that is currently an error, but may work in a future version
of Rust:

```compile_fail,E0107
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

```compile_fail,E0116
impl Vec<u8> { } // error
```

To fix this problem, you can do either of these things:

 - define a trait that has the desired associated functions/types/constants and
   implement the trait for the type in question
 - define a new type wrapping the type and define an implementation on the new
   type

Note that using the `type` keyword does not work here because `type` only
introduces a type alias:

```compile_fail,E0116
type Bytes = Vec<u8>;

impl Bytes { } // error, same as above
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

```compile_fail,E0117
impl Drop for u32 {}
```

To avoid this kind of error, ensure that at least one local type is referenced
by the `impl`:

```ignore
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
You're trying to write an inherent implementation for something which isn't a
struct nor an enum. Erroneous code example:

```compile_fail,E0118
impl (u8, u8) { // error: no base type found for inherent implementation
    fn get_state(&self) -> String {
        // ...
    }
}
```

To fix this error, please implement a trait on the type or wrap it in a struct.
Example:

```
// we create a trait here
trait LiveLongAndProsper {
    fn get_state(&self) -> String;
}

// and now you can implement it on (u8, u8)
impl LiveLongAndProsper for (u8, u8) {
    fn get_state(&self) -> String {
        "He's dead, Jim!".to_owned()
    }
}
```

Alternatively, you can create a newtype. A newtype is a wrapping tuple-struct.
For example, `NewType` is a newtype over `Foo` in `struct NewType(Foo)`.
Example:

```
struct TypeWrapper((u8, u8));

impl TypeWrapper {
    fn get_state(&self) -> String {
        "Fascinating!".to_owned()
    }
}
```
"##,

E0119: r##"
There are conflicting trait implementations for the same type.
Example of erroneous code:

```compile_fail,E0119
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
trait MyTrait {
    fn get(&self) -> usize;
}

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

```compile_fail,E0120
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

```compile_fail,E0121
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

```compile_fail,E0124
struct Foo {
    field1: i32,
    field1: i32, // error: field is already declared
}
```

Please verify that the field names have been correctly spelled. Example:

```
struct Foo {
    field1: i32,
    field2: i32, // ok!
}
```
"##,

E0128: r##"
Type parameter defaults can only use parameters that occur before them.
Erroneous code example:

```compile_fail,E0128
struct Foo<T=U, U=()> {
    field1: T,
    filed2: U,
}
// error: type parameters with a default cannot use forward declared
// identifiers
```

Since type parameters are evaluated in-order, you may be able to fix this issue
by doing:

```
struct Foo<U=(), T=U> {
    field1: T,
    filed2: U,
}
```

Please also verify that this wasn't because of a name-clash and rename the type
parameter if so.
"##,

E0131: r##"
It is not possible to define `main` with type parameters, or even with function
parameters. When `main` is present, it must take no arguments and return `()`.
Erroneous code example:

```compile_fail,E0131
fn main<T>() { // error: main function is not allowed to have type parameters
}
```
"##,

E0132: r##"
A function with the `start` attribute was declared with type parameters.

Erroneous code example:

```compile_fail,E0132
#![feature(start)]

#[start]
fn f<T>() {}
```

It is not possible to declare type parameters on a function that has the `start`
attribute. Such a function must have the following type signature (for more
information: http://doc.rust-lang.org/stable/book/no-stdlib.html):

```ignore
fn(isize, *const *const u8) -> isize;
```

Example:

```
#![feature(start)]

#[start]
fn my_start(argc: isize, argv: *const *const u8) -> isize {
    0
}
```
"##,

E0164: r##"
This error means that an attempt was made to match a struct type enum
variant as a non-struct type:

```compile_fail,E0164
enum Foo { B { i: u32 } }

fn bar(foo: Foo) -> u32 {
    match foo {
        Foo::B(i) => i, // error E0164
    }
}
```

Try using `{}` instead:

```
enum Foo { B { i: u32 } }

fn bar(foo: Foo) -> u32 {
    match foo {
        Foo::B{i} => i,
    }
}
```
"##,

E0182: r##"
You bound an associated type in an expression path which is not
allowed.

Erroneous code example:

```compile_fail,E0182
trait Foo {
    type A;
    fn bar() -> isize;
}

impl Foo for isize {
    type A = usize;
    fn bar() -> isize { 42 }
}

// error: unexpected binding of associated item in expression path
let x: isize = Foo::<A=usize>::bar();
```

To give a concrete type when using the Universal Function Call Syntax,
use "Type as Trait". Example:

```
trait Foo {
    type A;
    fn bar() -> isize;
}

impl Foo for isize {
    type A = usize;
    fn bar() -> isize { 42 }
}

let x: isize = <isize as Foo>::bar(); // ok!
```
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

```compile_fail,E0185
trait Foo {
    fn foo();
}

struct Bar;

impl Foo for Bar {
    // error, method `foo` has a `&self` declaration in the impl, but not in
    // the trait
    fn foo(&self) {}
}
```
"##,

E0186: r##"
An associated function for a trait was defined to be a method (i.e. to take a
`self` parameter), but an implementation of the trait declared the same function
to be static.

Here's an example of this error:

```compile_fail,E0186
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

```compile_fail,E0191
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

```ignore
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
trait Foo {
    fn bar(&self);
}

#[derive(Copy,Clone)]
struct Wrapper<T> {
    Wrapped: T
}
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

```compile_fail,E0194
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

```compile_fail,E0195
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

```
#![feature(optin_builtin_traits)]

struct Foo;

trait Enterprise {}

impl Enterprise for .. { }

impl !Enterprise for Foo { }
```

Please note that negative impls are only allowed for traits with default impls.
"##,

E0199: r##"
Safe traits should not have unsafe implementations, therefore marking an
implementation for a safe trait unsafe will cause a compiler error. Removing
the unsafe marker on the trait noted in the error will resolve this problem.

```compile_fail,E0199
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

```compile_fail,E0200
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

```compile_fail,E0201
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

Note, however, that items with the same name are allowed for inherent `impl`
blocks that don't overlap:

```
struct Foo<T>(T);

impl Foo<u8> {
    fn bar(&self) -> bool { self.0 > 5 }
}

impl Foo<bool> {
    fn bar(&self) -> bool { self.0 }
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

```compile_fail,E0204
struct Foo {
    foo : Vec<u32>,
}

impl Copy for Foo { }
```

This fails because `Vec<T>` does not implement `Copy` for any `T`.

Here's another example that will fail:

```compile_fail,E0204
#[derive(Copy)]
struct Foo<'a> {
    ty: &'a mut bool,
}
```

This fails because `&mut T` is not `Copy`, even when `T` is `Copy` (this
differs from the behavior for `&T`, which is always `Copy`).
"##,

/*
E0205: r##"
An attempt to implement the `Copy` trait for an enum failed because one of the
variants does not implement `Copy`. To fix this, you must implement `Copy` for
the mentioned variant. Note that this may not be possible, as in the example of

```compile_fail,E0205
enum Foo {
    Bar(Vec<u32>),
    Baz,
}

impl Copy for Foo { }
```

This fails because `Vec<T>` does not implement `Copy` for any `T`.

Here's another example that will fail:

```compile_fail,E0205
#[derive(Copy)]
enum Foo<'a> {
    Bar(&'a mut bool),
    Baz,
}
```

This fails because `&mut T` is not `Copy`, even when `T` is `Copy` (this
differs from the behavior for `&T`, which is always `Copy`).
"##,
*/

E0206: r##"
You can only implement `Copy` for a struct or enum. Both of the following
examples will fail, because neither `i32` (primitive type) nor `&'static Bar`
(reference to `Bar`) is a struct or enum:

```compile_fail,E0206
type Foo = i32;
impl Copy for Foo { } // error

#[derive(Copy, Clone)]
struct Bar;
impl Copy for &'static Bar { } // error
```
"##,

E0207: r##"
Any type parameter or lifetime parameter of an `impl` must meet at least one of
the following criteria:

 - it appears in the self type of the impl
 - for a trait impl, it appears in the trait reference
 - it is bound as an associated type

### Error example 1

Suppose we have a struct `Foo` and we would like to define some methods for it.
The following definition leads to a compiler error:

```compile_fail,E0207
struct Foo;

impl<T: Default> Foo {
// error: the type parameter `T` is not constrained by the impl trait, self
// type, or predicates [E0207]
    fn get(&self) -> T {
        <T as Default>::default()
    }
}
```

The problem is that the parameter `T` does not appear in the self type (`Foo`)
of the impl. In this case, we can fix the error by moving the type parameter
from the `impl` to the method `get`:


```
struct Foo;

// Move the type parameter from the impl to the method
impl Foo {
    fn get<T: Default>(&self) -> T {
        <T as Default>::default()
    }
}
```

### Error example 2

As another example, suppose we have a `Maker` trait and want to establish a
type `FooMaker` that makes `Foo`s:

```compile_fail,E0207
trait Maker {
    type Item;
    fn make(&mut self) -> Self::Item;
}

struct Foo<T> {
    foo: T
}

struct FooMaker;

impl<T: Default> Maker for FooMaker {
// error: the type parameter `T` is not constrained by the impl trait, self
// type, or predicates [E0207]
    type Item = Foo<T>;

    fn make(&mut self) -> Foo<T> {
        Foo { foo: <T as Default>::default() }
    }
}
```

This fails to compile because `T` does not appear in the trait or in the
implementing type.

One way to work around this is to introduce a phantom type parameter into
`FooMaker`, like so:

```
use std::marker::PhantomData;

trait Maker {
    type Item;
    fn make(&mut self) -> Self::Item;
}

struct Foo<T> {
    foo: T
}

// Add a type parameter to `FooMaker`
struct FooMaker<T> {
    phantom: PhantomData<T>,
}

impl<T: Default> Maker for FooMaker<T> {
    type Item = Foo<T>;

    fn make(&mut self) -> Foo<T> {
        Foo {
            foo: <T as Default>::default(),
        }
    }
}
```

Another way is to do away with the associated type in `Maker` and use an input
type parameter instead:

```
// Use a type parameter instead of an associated type here
trait Maker<Item> {
    fn make(&mut self) -> Item;
}

struct Foo<T> {
    foo: T
}

struct FooMaker;

impl<T: Default> Maker<Foo<T>> for FooMaker {
    fn make(&mut self) -> Foo<T> {
        Foo { foo: <T as Default>::default() }
    }
}
```

### Additional information

For more information, please see [RFC 447].

[RFC 447]: https://github.com/rust-lang/rfcs/blob/master/text/0447-no-unused-impl-parameters.md
"##,

E0210: r##"
This error indicates a violation of one of Rust's orphan rules for trait
implementations. The rule concerns the use of type parameters in an
implementation of a foreign trait (a trait defined in another crate), and
states that type parameters must be "covered" by a local type. To understand
what this means, it is perhaps easiest to consider a few examples.

If `ForeignTrait` is a trait defined in some external crate `foo`, then the
following trait `impl` is an error:

```compile_fail,E0210
extern crate collections;
use collections::range::RangeArgument;

impl<T> RangeArgument<T> for T { } // error

fn main() {}
```

To work around this, it can be covered with a local type, `MyType`:

```ignore
struct MyType<T>(T);
impl<T> ForeignTrait for MyType<T> { } // Ok
```

Please note that a type alias is not sufficient.

For another example of an error, suppose there's another trait defined in `foo`
named `ForeignTrait2` that takes two type parameters. Then this `impl` results
in the same rule violation:

```compile_fail
struct MyType2;
impl<T> ForeignTrait2<T, MyType<T>> for MyType2 { } // error
```

The reason for this is that there are two appearances of type parameter `T` in
the `impl` header, both as parameters for `ForeignTrait2`. The first appearance
is uncovered, and so runs afoul of the orphan rule.

Consider one more example:

```ignore
impl<T> ForeignTrait2<MyType<T>, T> for MyType2 { } // Ok
```

This only differs from the previous `impl` in that the parameters `T` and
`MyType<T>` for `ForeignTrait2` have been swapped. This example does *not*
violate the orphan rule; it is permitted.

To see why that last example was allowed, you need to understand the general
rule. Unfortunately this rule is a bit tricky to state. Consider an `impl`:

```ignore
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

/*
E0211: r##"
You used a function or type which doesn't fit the requirements for where it was
used. Erroneous code examples:

```compile_fail
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

```compile_fail
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
     */

E0214: r##"
A generic type was described using parentheses rather than angle brackets. For
example:

```compile_fail,E0214
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

```compile_fail,E0220
trait T1 {
    type Bar;
}

type Foo = T1<F=i32>; // error: associated type `F` not found for `T1`

// or:

trait T2 {
    type Bar;

    // error: Baz is used but not declared
    fn return_bool(&self, &Self::Bar, &Self::Baz) -> bool;
}
```

Make sure that you have defined the associated type in the trait body.
Also, verify that you used the right trait or you didn't misspell the
associated type name. Example:

```
trait T1 {
    type Bar;
}

type Foo = T1<Bar=i32>; // ok!

// or:

trait T2 {
    type Bar;
    type Baz; // we declare `Baz` in our trait.

    // and now we can use it here:
    fn return_bool(&self, &Self::Bar, &Self::Baz) -> bool;
}
```
"##,

E0221: r##"
An attempt was made to retrieve an associated type, but the type was ambiguous.
For example:

```compile_fail,E0221
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
trait T1 {}
trait T2 {}

trait Foo {
    type A: T1;
}

trait Bar : Foo {
    type A: T2;
    fn do_something() {
        let _: <Self as Bar>::A;
    }
}
```
"##,

E0223: r##"
An attempt was made to retrieve an associated type, but the type was ambiguous.
For example:

```compile_fail,E0223
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

```compile_fail,E0225
fn main() {
    let _: Box<std::io::Read + std::io::Write>;
}
```

Send and Sync are an exception to this rule: it's possible to have bounds of
one non-builtin trait, plus either or both of Send and Sync. For example, the
following compiles correctly:

```
fn main() {
    let _: Box<std::io::Read + Send + Sync>;
}
```
"##,

E0230: r##"
The trait has more type parameters specified than appear in its definition.

Erroneous example code:

```compile_fail,E0230
#![feature(on_unimplemented)]
#[rustc_on_unimplemented = "Trait error on `{Self}` with `<{A},{B},{C}>`"]
// error: there is no type parameter C on trait TraitWithThreeParams
trait TraitWithThreeParams<A,B>
{}
```

Include the correct number of type parameters and the compilation should
proceed:

```
#![feature(on_unimplemented)]
#[rustc_on_unimplemented = "Trait error on `{Self}` with `<{A},{B},{C}>`"]
trait TraitWithThreeParams<A,B,C> // ok!
{}
```
"##,

E0232: r##"
The attribute must have a value. Erroneous code example:

```compile_fail,E0232
#![feature(on_unimplemented)]

#[rustc_on_unimplemented] // error: this attribute must have a value
trait Bar {}
```

Please supply the missing value of the attribute. Example:

```
#![feature(on_unimplemented)]

#[rustc_on_unimplemented = "foo"] // ok!
trait Bar {}
```
"##,

E0243: r##"
This error indicates that not enough type parameters were found in a type or
trait.

For example, the `Foo` struct below is defined to be generic in `T`, but the
type parameter is missing in the definition of `Bar`:

```compile_fail,E0243
struct Foo<T> { x: T }

struct Bar { x: Foo }
```
"##,

E0244: r##"
This error indicates that too many type parameters were found in a type or
trait.

For example, the `Foo` struct below has no type parameters, but is supplied
with two in the definition of `Bar`:

```compile_fail,E0244
struct Foo { x: bool }

struct Bar<S, T> { x: Foo<S, T> }
```
"##,

E0569: r##"
If an impl has a generic parameter with the `#[may_dangle]` attribute, then
that impl must be declared as an `unsafe impl. For example:

```compile_fail,E0569
#![feature(generic_param_attrs)]
#![feature(dropck_eyepatch)]

struct Foo<X>(X);
impl<#[may_dangle] X> Drop for Foo<X> {
    fn drop(&mut self) { }
}
```

In this example, we are asserting that the destructor for `Foo` will not
access any data of type `X`, and require this assertion to be true for
overall safety in our program. The compiler does not currently attempt to
verify this assertion; therefore we must tag this `impl` as unsafe.
"##,

E0318: r##"
Default impls for a trait must be located in the same crate where the trait was
defined. For more information see the [opt-in builtin traits RFC](https://github
.com/rust-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md).
"##,

E0321: r##"
A cross-crate opt-out trait was implemented on something which wasn't a struct
or enum type. Erroneous code example:

```compile_fail,E0321
#![feature(optin_builtin_traits)]

struct Foo;

impl !Sync for Foo {}

unsafe impl Send for &'static Foo {}
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

```compile_fail,E0323
#![feature(associated_consts)]

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
```

Or:

```
#![feature(associated_consts)]

struct Bar;

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

```compile_fail,E0324
#![feature(associated_consts)]

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
#![feature(associated_consts)]

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

```compile_fail,E0325
#![feature(associated_consts)]

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
```

Or:

```
#![feature(associated_consts)]

struct Bar;

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

```compile_fail,E0326
#![feature(associated_consts)]

trait Foo {
    const BAR: bool;
}

struct Bar;

impl Foo for Bar {
    const BAR: u32 = 5; // error, expected bool, found u32
}
```
"##,

E0328: r##"
The Unsize trait should not be implemented directly. All implementations of
Unsize are provided automatically by the compiler.

Erroneous code example:

```compile_fail,E0328
#![feature(unsize)]

use std::marker::Unsize;

pub struct MyType;

impl<T> Unsize<T> for MyType {}
```

If you are defining your own smart pointer type and would like to enable
conversion from a sized to an unsized type with the [DST coercion system]
(https://github.com/rust-lang/rfcs/blob/master/text/0982-dst-coercion.md), use
[`CoerceUnsized`](https://doc.rust-lang.org/std/ops/trait.CoerceUnsized.html)
instead.

```
#![feature(coerce_unsized)]

use std::ops::CoerceUnsized;

pub struct MyType<T: ?Sized> {
    field_with_unsized_type: T,
}

impl<T, U> CoerceUnsized<MyType<U>> for MyType<T>
    where T: CoerceUnsized<U> {}
```
"##,

E0329: r##"
An attempt was made to access an associated constant through either a generic
type parameter or `Self`. This is not supported yet. An example causing this
error is shown below:

```ignore
#![feature(associated_consts)]

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

Currently, the value of `BAR` for a particular type can only be accessed
through a concrete type, as shown below:

```ignore
#![feature(associated_consts)]

trait Foo {
    const BAR: f64;
}

struct MyStruct;

fn get_bar_good() -> f64 {
    <MyStruct as Foo>::BAR
}
```
"##,

E0366: r##"
An attempt was made to implement `Drop` on a concrete specialization of a
generic type. An example is shown below:

```compile_fail,E0366
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

```compile_fail,E0367
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

```compile_fail,E0368
let mut x = 12f32; // error: binary operation `<<` cannot be applied to
                   //        type `f32`

x <<= 2;
```

To fix this error, please check that this type implements this binary
operation. Example:

```
let mut x = 12u32; // the `u32` type does implement the `ShlAssign` trait

x <<= 2; // ok!
```

It is also possible to overload most operators for your own type by
implementing the `[OP]Assign` traits from `std::ops`.

Another problem you might be facing is this: suppose you've overloaded the `+`
operator for some type `Foo` by implementing the `std::ops::Add` trait for
`Foo`, but you find that using `+=` does not work, as in this example:

```compile_fail,E0368
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

```compile_fail,E0369
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

```compile_fail
#[deny(overflowing_literals)]
enum Foo {
    X = 0x7fffffffffffffff,
    Y, // error: enum discriminant overflowed on value after
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
```

Or:

```
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

```compile_fail,E0371
trait Foo { fn foo(&self) { } }
trait Bar: Foo { }
trait Baz: Bar { }

impl Bar for Baz { } // error, `Baz` implements `Bar` by definition
impl Foo for Baz { } // error, `Baz` implements `Bar` which implements `Foo`
impl Baz for Baz { } // error, `Baz` (trivially) implements `Baz`
impl Baz for Bar { } // Note: This is OK
```
"##,

E0374: r##"
A struct without a field containing an unsized type cannot implement
`CoerceUnsized`. An
[unsized type](https://doc.rust-lang.org/book/unsized-types.html)
is any type that the compiler doesn't know the length or alignment of at
compile time. Any struct containing an unsized type is also unsized.

Example of erroneous code:

```compile_fail,E0374
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> {
    a: i32,
}

// error: Struct `Foo` has no unsized fields that need `CoerceUnsized`.
impl<T, U> CoerceUnsized<Foo<U>> for Foo<T>
    where T: CoerceUnsized<U> {}
```

`CoerceUnsized` is used to coerce one struct containing an unsized type
into another struct containing a different unsized type. If the struct
doesn't have any fields of unsized types then you don't need explicit
coercion to get the types you want. To fix this you can either
not try to implement `CoerceUnsized` or you can add a field that is
unsized to the struct.

Example:

```
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

// We don't need to impl `CoerceUnsized` here.
struct Foo {
    a: i32,
}

// We add the unsized type field to the struct.
struct Bar<T: ?Sized> {
    a: i32,
    b: T,
}

// The struct has an unsized field so we can implement
// `CoerceUnsized` for it.
impl<T, U> CoerceUnsized<Bar<U>> for Bar<T>
    where T: CoerceUnsized<U> {}
```

Note that `CoerceUnsized` is mainly used by smart pointers like `Box`, `Rc`
and `Arc` to be able to mark that they can coerce unsized types that they
are pointing at.
"##,

E0375: r##"
A struct with more than one field containing an unsized type cannot implement
`CoerceUnsized`. This only occurs when you are trying to coerce one of the
types in your struct to another type in the struct. In this case we try to
impl `CoerceUnsized` from `T` to `U` which are both types that the struct
takes. An [unsized type](https://doc.rust-lang.org/book/unsized-types.html)
is any type that the compiler doesn't know the length or alignment of at
compile time. Any struct containing an unsized type is also unsized.

Example of erroneous code:

```compile_fail,E0375
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized, U: ?Sized> {
    a: i32,
    b: T,
    c: U,
}

// error: Struct `Foo` has more than one unsized field.
impl<T, U> CoerceUnsized<Foo<U, T>> for Foo<T, U> {}
```

`CoerceUnsized` only allows for coercion from a structure with a single
unsized type field to another struct with a single unsized type field.
In fact Rust only allows for a struct to have one unsized type in a struct
and that unsized type must be the last field in the struct. So having two
unsized types in a single struct is not allowed by the compiler. To fix this
use only one field containing an unsized type in the struct and then use
multiple structs to manage each unsized type field you need.

Example:

```
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> {
    a: i32,
    b: T,
}

impl <T, U> CoerceUnsized<Foo<U>> for Foo<T>
    where T: CoerceUnsized<U> {}

fn coerce_foo<T: CoerceUnsized<U>, U>(t: T) -> Foo<U> {
    Foo { a: 12i32, b: t } // we use coercion to get the `Foo<U>` type we need
}
```

"##,

E0376: r##"
The type you are trying to impl `CoerceUnsized` for is not a struct.
`CoerceUnsized` can only be implemented for a struct. Unsized types are
already able to be coerced without an implementation of `CoerceUnsized`
whereas a struct containing an unsized type needs to know the unsized type
field it's containing is able to be coerced. An
[unsized type](https://doc.rust-lang.org/book/unsized-types.html)
is any type that the compiler doesn't know the length or alignment of at
compile time. Any struct containing an unsized type is also unsized.

Example of erroneous code:

```compile_fail,E0376
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> {
    a: T,
}

// error: The type `U` is not a struct
impl<T, U> CoerceUnsized<U> for Foo<T> {}
```

The `CoerceUnsized` trait takes a struct type. Make sure the type you are
providing to `CoerceUnsized` is a struct with only the last field containing an
unsized type.

Example:

```
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T> {
    a: T,
}

// The `Foo<U>` is a struct so `CoerceUnsized` can be implemented
impl<T, U> CoerceUnsized<Foo<U>> for Foo<T> where T: CoerceUnsized<U> {}
```

Note that in Rust, structs can only contain an unsized type if the field
containing the unsized type is the last and only unsized type field in the
struct.
"##,

E0380: r##"
Default impls are only allowed for traits with no methods or associated items.
For more information see the [opt-in builtin traits RFC](https://github.com/rust
-lang/rfcs/blob/master/text/0019-opt-in-builtin-traits.md).
"##,

E0390: r##"
You tried to implement methods for a primitive type. Erroneous code example:

```compile_fail,E0390
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

```compile_fail,E0391
trait FirstTrait : SecondTrait {

}

trait SecondTrait : FirstTrait {

}
```
"##,

E0392: r##"
This error indicates that a type or lifetime parameter has been declared
but not actually used. Here is an example that demonstrates the error:

```compile_fail,E0392
enum Foo<T> {
    Bar,
}
```

If the type parameter was included by mistake, this error can be fixed
by simply removing the type parameter, as shown below:

```
enum Foo {
    Bar,
}
```

Alternatively, if the type parameter was intentionally inserted, it must be
used. A simple fix is shown below:

```
enum Foo<T> {
    Bar(T),
}
```

This error may also commonly be found when working with unsafe code. For
example, when using raw pointers one may wish to specify the lifetime for
which the pointed-at data is valid. An initial attempt (below) causes this
error:

```compile_fail,E0392
struct Foo<'a, T> {
    x: *const T,
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

E0393: r##"
A type parameter which references `Self` in its default value was not specified.
Example of erroneous code:

```compile_fail,E0393
trait A<T=Self> {}

fn together_we_will_rule_the_galaxy(son: &A) {}
// error: the type parameter `T` must be explicitly specified in an
//        object type because its default value `Self` references the
//        type `Self`
```

A trait object is defined over a single, fully-defined trait. With a regular
default parameter, this parameter can just be substituted in. However, if the
default parameter is `Self`, the trait changes for each concrete type; i.e.
`i32` will be expected to implement `A<i32>`, `bool` will be expected to
implement `A<bool>`, etc... These types will not share an implementation of a
fully-defined trait; instead they share implementations of a trait with
different parameters substituted in for each implementation. This is
irreconcilable with what we need to make a trait object work, and is thus
disallowed. Making the trait concrete by explicitly specifying the value of the
defaulted parameter will fix this issue. Fixed example:

```
trait A<T=Self> {}

fn together_we_will_rule_the_galaxy(son: &A<i32>) {} // Ok!
```
"##,

E0399: r##"
You implemented a trait, overriding one or more of its associated types but did
not reimplement its default methods.

Example of erroneous code:

```compile_fail,E0399
#![feature(associated_type_defaults)]

pub trait Foo {
    type Assoc = u8;
    fn bar(&self) {}
}

impl Foo for i32 {
    // error - the following trait items need to be reimplemented as
    //         `Assoc` was overridden: `bar`
    type Assoc = i32;
}
```

To fix this, add an implementation for each default method from the trait:

```
#![feature(associated_type_defaults)]

pub trait Foo {
    type Assoc = u8;
    fn bar(&self) {}
}

impl Foo for i32 {
    type Assoc = i32;
    fn bar(&self) {} // ok!
}
```
"##,

E0439: r##"
The length of the platform-intrinsic function `simd_shuffle`
wasn't specified. Erroneous code example:

```compile_fail,E0439
#![feature(platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_shuffle<A,B>(a: A, b: A, c: [u32; 8]) -> B;
    // error: invalid `simd_shuffle`, needs length: `simd_shuffle`
}
```

The `simd_shuffle` function needs the length of the array passed as
last parameter in its name. Example:

```
#![feature(platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_shuffle8<A,B>(a: A, b: A, c: [u32; 8]) -> B;
}
```
"##,

E0440: r##"
A platform-specific intrinsic function has the wrong number of type
parameters. Erroneous code example:

```compile_fail,E0440
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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

```compile_fail,E0441
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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

```compile_fail,E0442
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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

```compile_fail,E0443
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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

```compile_fail,E0444
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

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

```compile_fail,E0516
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

E0520: r##"
A non-default implementation was already made on this type so it cannot be
specialized further. Erroneous code example:

```compile_fail,E0520
#![feature(specialization)]

trait SpaceLlama {
    fn fly(&self);
}

// applies to all T
impl<T> SpaceLlama for T {
    default fn fly(&self) {}
}

// non-default impl
// applies to all `Clone` T and overrides the previous impl
impl<T: Clone> SpaceLlama for T {
    fn fly(&self) {}
}

// since `i32` is clone, this conflicts with the previous implementation
impl SpaceLlama for i32 {
    default fn fly(&self) {}
    // error: item `fly` is provided by an `impl` that specializes
    //        another, but the item in the parent `impl` is not marked
    //        `default` and so it cannot be specialized.
}
```

Specialization only allows you to override `default` functions in
implementations.

To fix this error, you need to mark all the parent implementations as default.
Example:

```
#![feature(specialization)]

trait SpaceLlama {
    fn fly(&self);
}

// applies to all T
impl<T> SpaceLlama for T {
    default fn fly(&self) {} // This is a parent implementation.
}

// applies to all `Clone` T; overrides the previous impl
impl<T: Clone> SpaceLlama for T {
    default fn fly(&self) {} // This is a parent implementation but was
                             // previously not a default one, causing the error
}

// applies to i32, overrides the previous two impls
impl SpaceLlama for i32 {
    fn fly(&self) {} // And now that's ok!
}
```
"##,

E0527: r##"
The number of elements in an array or slice pattern differed from the number of
elements in the array being matched.

Example of erroneous code:

```compile_fail,E0527
#![feature(slice_patterns)]

let r = &[1, 2, 3, 4];
match r {
    &[a, b] => { // error: pattern requires 2 elements but array
                 //        has 4
        println!("a={}, b={}", a, b);
    }
}
```

Ensure that the pattern is consistent with the size of the matched
array. Additional elements can be matched with `..`:

```
#![feature(slice_patterns)]

let r = &[1, 2, 3, 4];
match r {
    &[a, b, ..] => { // ok!
        println!("a={}, b={}", a, b);
    }
}
```
"##,

E0528: r##"
An array or slice pattern required more elements than were present in the
matched array.

Example of erroneous code:

```compile_fail,E0528
#![feature(slice_patterns)]

let r = &[1, 2];
match r {
    &[a, b, c, rest..] => { // error: pattern requires at least 3
                            //        elements but array has 2
        println!("a={}, b={}, c={} rest={:?}", a, b, c, rest);
    }
}
```

Ensure that the matched array has at least as many elements as the pattern
requires. You can match an arbitrary number of remaining elements with `..`:

```
#![feature(slice_patterns)]

let r = &[1, 2, 3, 4, 5];
match r {
    &[a, b, c, rest..] => { // ok!
        // prints `a=1, b=2, c=3 rest=[4, 5]`
        println!("a={}, b={}, c={} rest={:?}", a, b, c, rest);
    }
}
```
"##,

E0529: r##"
An array or slice pattern was matched against some other type.

Example of erroneous code:

```compile_fail,E0529
#![feature(slice_patterns)]

let r: f32 = 1.0;
match r {
    [a, b] => { // error: expected an array or slice, found `f32`
        println!("a={}, b={}", a, b);
    }
}
```

Ensure that the pattern and the expression being matched on are of consistent
types:

```
#![feature(slice_patterns)]

let r = [1.0, 2.0];
match r {
    [a, b] => { // ok!
        println!("a={}, b={}", a, b);
    }
}
```
"##,

E0559: r##"
An unknown field was specified into an enum's structure variant.

Erroneous code example:

```compile_fail,E0559
enum Field {
    Fool { x: u32 },
}

let s = Field::Fool { joke: 0 };
// error: struct variant `Field::Fool` has no field named `joke`
```

Verify you didn't misspell the field's name or that the field exists. Example:

```
enum Field {
    Fool { joke: u32 },
}

let s = Field::Fool { joke: 0 }; // ok!
```
"##,

E0560: r##"
An unknown field was specified into a structure.

Erroneous code example:

```compile_fail,E0560
struct Simba {
    mother: u32,
}

let s = Simba { mother: 1, father: 0 };
// error: structure `Simba` has no field named `father`
```

Verify you didn't misspell the field's name or that the field exists. Example:

```
struct Simba {
    mother: u32,
    father: u32,
}

let s = Simba { mother: 1, father: 0 }; // ok!
```
"##,

E0570: r##"
The requested ABI is unsupported by the current target.

The rust compiler maintains for each target a blacklist of ABIs unsupported on
that target. If an ABI is present in such a list this usually means that the
target / ABI combination is currently unsupported by llvm.

If necessary, you can circumvent this check using custom target specifications.
"##,

E0572: r##"
A return statement was found outside of a function body.

Erroneous code example:

```compile_fail,E0572
const FOO: u32 = return 0; // error: return statement outside of function body

fn main() {}
```

To fix this issue, just remove the return keyword or move the expression into a
function. Example:

```
const FOO: u32 = 0;

fn some_fn() -> u32 {
    return FOO;
}

fn main() {
    some_fn();
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
//  E0163, // merged into E0071
//  E0167,
//  E0168,
//  E0172, // non-trait found in a type sum, moved to resolve
//  E0173, // manual implementations of unboxed closure traits are experimental
//  E0174,
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
    E0231, // only named substitution parameters are allowed
//  E0233,
//  E0234,
//  E0235, // structure constructor specifies a structure of type but
//  E0236, // no lang item for range syntax
//  E0237, // no lang item for range syntax
//  E0238, // parenthesized parameters may only be used with a trait
//  E0239, // `next` method of `Iterator` trait has unexpected type
//  E0240,
//  E0241,
//  E0242,
    E0245, // not a trait
//  E0246, // invalid recursive type
//  E0247,
//  E0248, // value used as a type, now reported earlier during resolution as E0412
//  E0249,
//  E0319, // trait impls for defaulted traits allowed just for structs/enums
    E0320, // recursive overflow during dropck
//  E0372, // coherence not object safe
    E0377, // the trait `CoerceUnsized` may only be implemented for a coercion
           // between structures with the same definition
    E0436, // functional record update requires a struct
    E0521, // redundant default implementations of trait
    E0533, // `{}` does not name a unit variant, unit struct or a constant
    E0562, // `impl Trait` not allowed outside of function
           // and inherent method return types
    E0563, // cannot determine a type for this `impl Trait`: {}
    E0564, // only named lifetimes are allowed in `impl Trait`,
           // but `{}` was found in the type `{}`
    E0567, // auto traits can not have type parameters
    E0568, // auto-traits can not have predicates,
}
