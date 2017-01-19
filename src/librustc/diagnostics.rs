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
E0020: r##"
This error indicates that an attempt was made to divide by zero (or take the
remainder of a zero divisor) in a static or constant expression. Erroneous
code example:

```compile_fail
#[deny(const_err)]

const X: i32 = 42 / 0;
// error: attempt to divide by zero in a constant expression
```
"##,

E0038: r##"
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

We cannot create an object of type `Box<Foo>` or `&Foo` since in this case
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
cause this problem.)

In such a case, the compiler cannot predict the return type of `foo()` in a
situation like the following:

```compile_fail
trait Trait {
    fn foo(&self) -> Self;
}

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
methods. With such a bound, one can still call `foo()` on types implementing
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

The machine code for `foo::<u8>()`, `foo::<bool>()`, `foo::<String>()`, or any
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

```ignore
fn call_foo(thing: Box<Trait>) {
    thing.foo(true); // this could be any one of the 8 types above
    thing.foo(1);
    thing.foo("hello");
}
```

We don't just need to create a table of all implementations of all methods of
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
a way to get a pointer to the method table for them.

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

```compile_fail
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
"##,

E0072: r##"
When defining a recursive struct or enum, any use of the type being defined
from inside the definition must occur behind a pointer (like `Box` or `&`).
This is because structs and enums must have a well-defined size, and without
the pointer, the size of the type would need to be unbounded.

Consider the following erroneous definition of a type for a list of bytes:

```compile_fail,E0072
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

E0109: r##"
You tried to give a type parameter to a type which doesn't need it. Erroneous
code example:

```compile_fail,E0109
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

```compile_fail,E0110
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
Unsafe code was used outside of an unsafe function or block.

Erroneous code example:

```compile_fail,E0133
unsafe fn f() { return; } // This is the unsafe code

fn main() {
    f(); // error: call to unsafe function requires unsafe function or block
}
```

Using unsafe functionality is potentially dangerous and disallowed by safety
checks. Examples:

* Dereferencing raw pointers
* Calling functions via FFI
* Calling functions marked unsafe

These safety checks can be relaxed for a section of the code by wrapping the
unsafe instructions with an `unsafe` block. For instance:

```
unsafe fn f() { return; }

fn main() {
    unsafe { f(); } // ok!
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
More than one function was declared with the `#[main]` attribute.

Erroneous code example:

```compile_fail,E0137
#![feature(main)]

#[main]
fn foo() {}

#[main]
fn f() {} // error: multiple functions with a #[main] attribute
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

// isn't thrown anymore
E0139: r##"
There are various restrictions on transmuting between types in Rust; for example
types being transmuted must have the same size. To apply all these restrictions,
the compiler must know the exact types that may be transmuted. When type
parameters are involved, this cannot always be done.

So, for example, the following is not allowed:

```
use std::mem::transmute;

struct Foo<T>(Vec<T>);

fn foo<T>(x: Vec<T>) {
    // we are transmuting between Vec<T> and Foo<F> here
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

```ignore
struct Foo<T>(Vec<T>);

trait MyTransmutableType {
    fn transmute(Vec<Self>) -> Foo<Self>;
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

```ignore
ptr::read(&v as *const _ as *const SomeType) // `v` transmuted to `SomeType`
```

Note that this does not move `v` (unlike `transmute`), and may need a
call to `mem::forget(v)` in case you want to avoid destructors being called.
"##,

E0152: r##"
A lang item was redefined.

Erroneous code example:

```compile_fail,E0152
#![feature(lang_items)]

#[lang = "panic_fmt"]
struct Foo; // error: duplicate lang item found: `panic_fmt`
```

Lang items are already implemented in the standard library. Unless you are
writing a free-standing application (e.g. a kernel), you do not need to provide
them yourself.

You can build a free-standing crate by adding `#![no_std]` to the crate
attributes:

```
#![no_std]
```

See also https://doc.rust-lang.org/book/no-stdlib.html
"##,

E0229: r##"
An associated type binding was done outside of the type parameter declaration
and `where` clause. Erroneous code example:

```compile_fail,E0229
pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

struct Bar;

impl Foo for isize {
    type A = usize;
    fn boo(&self) -> usize { 42 }
}

fn baz<I>(x: &<I as Foo<A=Bar>>::A) {}
// error: associated type bindings are not allowed here
```

To solve this error, please move the type bindings in the type parameter
declaration:

```ignore
fn baz<I: Foo<A=Bar>>(x: &<I as Foo>::A) {} // ok!
```

Or in the `where` clause:

```ignore
fn baz<I>(x: &<I as Foo>::A) where I: Foo<A=Bar> {}
```
"##,

E0261: r##"
When using a lifetime like `'a` in a type, it must be declared before being
used.

These two examples illustrate the problem:

```compile_fail,E0261
// error, use of undeclared lifetime name `'a`
fn foo(x: &'a str) { }

struct Foo {
    // error, use of undeclared lifetime name `'a`
    x: &'a str,
}
```

These can be fixed by declaring lifetime parameters:

```
fn foo<'a>(x: &'a str) {}

struct Foo<'a> {
    x: &'a str,
}
```
"##,

E0262: r##"
Declaring certain lifetime names in parameters is disallowed. For example,
because the `'static` lifetime is a special built-in lifetime name denoting
the lifetime of the entire program, this is an error:

```compile_fail,E0262
// error, invalid lifetime parameter name `'static`
fn foo<'static>(x: &'static str) { }
```
"##,

E0263: r##"
A lifetime name cannot be declared more than once in the same scope. For
example:

```compile_fail,E0263
// error, lifetime name `'a` declared twice in the same scope
fn foo<'a, 'b, 'a>(x: &'a str, y: &'b str) { }
```
"##,

E0264: r##"
An unknown external lang item was used. Erroneous code example:

```compile_fail,E0264
#![feature(lang_items)]

extern "C" {
    #[lang = "cake"] // error: unknown external lang item: `cake`
    fn cake();
}
```

A list of available external lang items is available in
`src/librustc/middle/weak_lang_items.rs`. Example:

```
#![feature(lang_items)]

extern "C" {
    #[lang = "panic_fmt"] // ok!
    fn cake();
}
```
"##,

E0271: r##"
This is because of a type mismatch between the associated type of some
trait (e.g. `T::Bar`, where `T` implements `trait Quux { type Bar; }`)
and another type `U` that is required to be equal to `T::Bar`, but is not.
Examples follow.

Here is a basic example:

```compile_fail,E0271
trait Trait { type AssociatedType; }

fn foo<T>(t: T) where T: Trait<AssociatedType=u32> {
    println!("in foo");
}

impl Trait for i8 { type AssociatedType = &'static str; }

foo(3_i8);
```

Here is that same example again, with some explanatory comments:

```ignore
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

```compile_fail
let vs: Vec<i32> = vec![1, 2, 3, 4];
for v in &vs {
    match v {
        1 => {},
        _ => {},
    }
}
```

The above fails because of an analogous type mismatch,
though may be harder to see. Again, here are some
explanatory comments for the same example:

```ignore
{
    let vs = vec![1, 2, 3, 4];

    // `for`-loops use a protocol based on the `Iterator`
    // trait. Each item yielded in a `for` loop has the
    // type `Iterator::Item` -- that is, `Item` is the
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

```compile_fail
#![feature(on_unimplemented)]

fn foo<T: Index<u8>>(x: T){}

#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
trait Index<Idx> { /* ... */ }

foo(true); // `bool` does not implement `Index<u8>`
```

There will be an error about `bool` not implementing `Index<u8>`, followed by a
note saying "the type `bool` cannot be indexed by `u8`".

As you can see, you can specify type parameters in curly braces for
substitution with the actual types (using the regular format string syntax) in
a given situation. Furthermore, `{Self}` will substitute to the type (in this
case, `bool`) that we tried to use.

This error appears when the curly braces contain an identifier which doesn't
match with any of the type parameters or the string `Self`. This might happen
if you misspelled a type parameter, or if you intended to use literal curly
braces. If it is the latter, escape the curly braces with a second curly brace
of the same type; e.g. a literal `{` is `{{`.
"##,

E0273: r##"
The `#[rustc_on_unimplemented]` attribute lets you specify a custom error
message for when a particular trait isn't implemented on a type placed in a
position that needs that trait. For example, when the following code is
compiled:

```compile_fail
#![feature(on_unimplemented)]

fn foo<T: Index<u8>>(x: T){}

#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
trait Index<Idx> { /* ... */ }

foo(true); // `bool` does not implement `Index<u8>`
```

there will be an error about `bool` not implementing `Index<u8>`, followed by a
note saying "the type `bool` cannot be indexed by `u8`".

As you can see, you can specify type parameters in curly braces for
substitution with the actual types (using the regular format string syntax) in
a given situation. Furthermore, `{Self}` will substitute to the type (in this
case, `bool`) that we tried to use.

This error appears when the curly braces do not contain an identifier. Please
add one of the same name as a type parameter. If you intended to use literal
braces, use `{{` and `}}` to escape them.
"##,

E0274: r##"
The `#[rustc_on_unimplemented]` attribute lets you specify a custom error
message for when a particular trait isn't implemented on a type placed in a
position that needs that trait. For example, when the following code is
compiled:

```compile_fail
#![feature(on_unimplemented)]

fn foo<T: Index<u8>>(x: T){}

#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
trait Index<Idx> { /* ... */ }

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
before it could be evaluated. Often this means that there is unbounded
recursion in resolving some type bounds.

For example, in the following code:

```compile_fail,E0275
trait Foo {}

struct Bar<T>(T);

impl<T> Foo for T where Bar<T>: Foo {}
```

To determine if a `T` is `Foo`, we need to check if `Bar<T>` is `Foo`. However,
to do this check, we need to determine that `Bar<Bar<T>>` is `Foo`. To
determine this, we check if `Bar<Bar<Bar<T>>>` is `Foo`, and so on. This is
clearly a recursive requirement that can't be resolved directly.

Consider changing your trait bounds so that they're less self-referential.
"##,

E0276: r##"
This error occurs when a bound in an implementation of a trait does not match
the bounds specified in the original trait. For example:

```compile_fail,E0276
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

```compile_fail,E0277
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
    some_func(5i32); // error: the trait bound `i32 : Foo` is not satisfied
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

Or in a generic context, an erroneous code example would look like:

```compile_fail,E0277
fn some_func<T>(foo: T) {
    println!("{:?}", foo); // error: the trait `core::fmt::Debug` is not
                           //        implemented for the type `T`
}

fn main() {
    // We now call the method with the i32 type,
    // which *does* implement the Debug trait.
    some_func(5i32);
}
```

Note that the error here is in the definition of the generic function: Although
we only call it with a parameter that does implement `Debug`, the compiler
still rejects the function: It must work with all possible input types. In
order to make this example compile, we need to restrict the generic type we're
accepting:

```
use std::fmt;

// Restrict the input type to types that implement Debug.
fn some_func<T: fmt::Debug>(foo: T) {
    println!("{:?}", foo);
}

fn main() {
    // Calling the method is still fine, as i32 implements Debug.
    some_func(5i32);

    // This would fail to compile now:
    // struct WithoutDebug;
    // some_func(WithoutDebug);
}
```

Rust only looks at the signature of the called function, as such it must
already specify all requirements that will be used for every type parameter.
"##,

E0281: r##"
You tried to supply a type which doesn't implement some trait in a location
which expected that trait. This error typically occurs when working with
`Fn`-based types. Erroneous code example:

```compile_fail,E0281
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

```compile_fail,E0282
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

```compile_fail,E0282
struct Foo<T> {
    num: T,
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

E0283: r##"
This error occurs when the compiler doesn't have enough information
to unambiguously choose an implementation.

For example:

```compile_fail,E0283
trait Generator {
    fn create() -> u32;
}

struct Impl;

impl Generator for Impl {
    fn create() -> u32 { 1 }
}

struct AnotherImpl;

impl Generator for AnotherImpl {
    fn create() -> u32 { 2 }
}

fn main() {
    let cont: u32 = Generator::create();
    // error, impossible to choose one of Generator trait implementation
    // Impl or AnotherImpl? Maybe anything else?
}
```

To resolve this error use the concrete type:

```
trait Generator {
    fn create() -> u32;
}

struct AnotherImpl;

impl Generator for AnotherImpl {
    fn create() -> u32 { 2 }
}

fn main() {
    let gen1 = AnotherImpl::create();

    // if there are multiple methods with same name (different traits)
    let gen2 = <AnotherImpl as Generator>::create();
}
```
"##,

E0296: r##"
This error indicates that the given recursion limit could not be parsed. Ensure
that the value provided is a positive integer between quotes.

Erroneous code example:

```compile_fail,E0296
#![recursion_limit]

fn main() {}
```

And a working example:

```
#![recursion_limit="1000"]

fn main() {}
```
"##,

E0308: r##"
This error occurs when the compiler was unable to infer the concrete type of a
variable. It can occur for several cases, the most common of which is a
mismatch in the expected type that the compiler inferred for a variable's
initializing expression, and the actual type explicitly assigned to the
variable.

For example:

```compile_fail,E0308
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

```compile_fail,E0309
// This won't compile because T is not constrained, meaning the data
// stored in it is not guaranteed to last as long as the reference
struct Foo<'a, T> {
    foo: &'a T
}
```

This will compile, because it has the constraint on the type parameter:

```
struct Foo<'a, T: 'a> {
    foo: &'a T
}
```

To see why this is important, consider the case where `T` is itself a reference
(e.g., `T = &str`). If we don't include the restriction that `T: 'a`, the
following code would be perfectly legal:

```compile_fail,E0309
struct Foo<'a, T> {
    foo: &'a T
}

fn main() {
    let v = "42".to_string();
    let f = Foo{foo: &v};
    drop(v);
    println!("{}", f.foo); // but we've already dropped v!
}
```
"##,

E0310: r##"
Types in type definitions have lifetimes associated with them that represent
how long the data stored within them is guaranteed to be live. This lifetime
must be as long as the data needs to be alive, and missing the constraint that
denotes this will cause this error.

```compile_fail,E0310
// This won't compile because T is not constrained to the static lifetime
// the reference needs
struct Foo<T> {
    foo: &'static T
}
```

This will compile, because it has the constraint on the type parameter:

```
struct Foo<T: 'static> {
    foo: &'static T
}
```
"##,

E0312: r##"
A lifetime of reference outlives lifetime of borrowed content.

Erroneous code example:

```compile_fail,E0312
fn make_child<'human, 'elve>(x: &mut &'human isize, y: &mut &'elve isize) {
    *x = *y;
    // error: lifetime of reference outlives lifetime of borrowed content
}
```

The compiler cannot determine if the `human` lifetime will live long enough
to keep up on the elve one. To solve this error, you have to give an
explicit lifetime hierarchy:

```
fn make_child<'human, 'elve: 'human>(x: &mut &'human isize,
                                     y: &mut &'elve isize) {
    *x = *y; // ok!
}
```

Or use the same lifetime for every variable:

```
fn make_child<'elve>(x: &mut &'elve isize, y: &mut &'elve isize) {
    *x = *y; // ok!
}
```
"##,

E0317: r##"
This error occurs when an `if` expression without an `else` block is used in a
context where a type other than `()` is expected, for example a `let`
expression:

```compile_fail,E0317
fn main() {
    let x = 5;
    let a = if x == 5 { 1 };
}
```

An `if` expression without an `else` block has the type `()`, so this is a type
error. To resolve it, add an `else` block having the same type as the `if`
block.
"##,

E0398: r##"
In Rust 1.3, the default object lifetime bounds are expected to change, as
described in RFC #1156 [1]. You are getting a warning because the compiler
thinks it is possible that this change will cause a compilation error in your
code. It is possible, though unlikely, that this is a false alarm.

The heart of the change is that where `&'a Box<SomeTrait>` used to default to
`&'a Box<SomeTrait+'a>`, it now defaults to `&'a Box<SomeTrait+'static>` (here,
`SomeTrait` is the name of some trait type). Note that the only types which are
affected are references to boxes, like `&Box<SomeTrait>` or
`&[Box<SomeTrait>]`. More common types like `&SomeTrait` or `Box<SomeTrait>`
are unaffected.

To silence this warning, edit your code to use an explicit bound. Most of the
time, this means that you will want to change the signature of a function that
you are calling. For example, if the error is reported on a call like `foo(x)`,
and `foo` is defined as follows:

```ignore
fn foo(arg: &Box<SomeTrait>) { ... }
```

You might change it to:

```ignore
fn foo<'a>(arg: &Box<SomeTrait+'a>) { ... }
```

This explicitly states that you expect the trait object `SomeTrait` to contain
references (with a maximum lifetime of `'a`).

[1]: https://github.com/rust-lang/rfcs/pull/1156
"##,

E0452: r##"
An invalid lint attribute has been given. Erroneous code example:

```compile_fail,E0452
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

E0453: r##"
A lint check attribute was overruled by a `forbid` directive set as an
attribute on an enclosing scope, or on the command line with the `-F` option.

Example of erroneous code:

```compile_fail,E0453
#![forbid(non_snake_case)]

#[allow(non_snake_case)]
fn main() {
    let MyNumber = 2; // error: allow(non_snake_case) overruled by outer
                      //        forbid(non_snake_case)
}
```

The `forbid` lint setting, like `deny`, turns the corresponding compiler
warning into a hard error. Unlike `deny`, `forbid` prevents itself from being
overridden by inner attributes.

If you're sure you want to override the lint check, you can change `forbid` to
`deny` (or use `-D` instead of `-F` if the `forbid` setting was given as a
command-line option) to allow the inner lint check attribute:

```
#![deny(non_snake_case)]

#[allow(non_snake_case)]
fn main() {
    let MyNumber = 2; // ok!
}
```

Otherwise, edit the code to pass the lint check, and remove the overruled
attribute:

```
#![forbid(non_snake_case)]

fn main() {
    let my_number = 2;
}
```
"##,

E0478: r##"
A lifetime bound was not satisfied.

Erroneous code example:

```compile_fail,E0478
// Check that the explicit lifetime bound (`'SnowWhite`, in this example) must
// outlive all the superbounds from the trait (`'kiss`, in this example).

trait Wedding<'t>: 't { }

struct Prince<'kiss, 'SnowWhite> {
    child: Box<Wedding<'kiss> + 'SnowWhite>,
    // error: lifetime bound not satisfied
}
```

In this example, the `'SnowWhite` lifetime is supposed to outlive the `'kiss`
lifetime but the declaration of the `Prince` struct doesn't enforce it. To fix
this issue, you need to specify it:

```
trait Wedding<'t>: 't { }

struct Prince<'kiss, 'SnowWhite: 'kiss> { // You say here that 'kiss must live
                                          // longer than 'SnowWhite.
    child: Box<Wedding<'kiss> + 'SnowWhite>, // And now it's all good!
}
```
"##,

E0491: r##"
A reference has a longer lifetime than the data it references.

Erroneous code example:

```compile_fail,E0491
// struct containing a reference requires a lifetime parameter,
// because the data the reference points to must outlive the struct (see E0106)
struct Struct<'a> {
    ref_i32: &'a i32,
}

// However, a nested struct like this, the signature itself does not tell
// whether 'a outlives 'b or the other way around.
// So it could be possible that 'b of reference outlives 'a of the data.
struct Nested<'a, 'b> {
    ref_struct: &'b Struct<'a>, // compile error E0491
}
```

To fix this issue, you can specify a bound to the lifetime like below:

```
struct Struct<'a> {
    ref_i32: &'a i32,
}

// 'a: 'b means 'a outlives 'b
struct Nested<'a: 'b, 'b> {
    ref_struct: &'b Struct<'a>,
}
```
"##,

E0496: r##"
A lifetime name is shadowing another lifetime name. Erroneous code example:

```compile_fail,E0496
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

```compile_fail
#[stable] // error: stability attributes may not be used outside of the
          //        standard library
fn foo() {}
```

It is not possible to use stability attributes outside of the standard library.
Also, for now, it is not possible to write deprecation messages either.
"##,

E0512: r##"
Transmute with two differently sized types was attempted. Erroneous code
example:

```compile_fail,E0512
fn takes_u8(_: u8) {}

fn main() {
    unsafe { takes_u8(::std::mem::transmute(0u16)); }
    // error: transmute called with differently sized types
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

E0517: r##"
This error indicates that a `#[repr(..)]` attribute was placed on an
unsupported item.

Examples of erroneous code:

```compile_fail,E0517
#[repr(C)]
type Foo = u8;

#[repr(packed)]
enum Foo {Bar, Baz}

#[repr(u8)]
struct Foo {bar: bool, baz: bool}

#[repr(C)]
impl Foo {
    // ...
}
```

* The `#[repr(C)]` attribute can only be placed on structs and enums.
* The `#[repr(packed)]` and `#[repr(simd)]` attributes only work on structs.
* The `#[repr(u8)]`, `#[repr(i16)]`, etc attributes only work on enums.

These attributes do not work on typedefs, since typedefs are just aliases.

Representations like `#[repr(u8)]`, `#[repr(i64)]` are for selecting the
discriminant size for C-like enums (when there is no associated data, e.g.
`enum Color {Red, Blue, Green}`), effectively setting the size of the enum to
the size of the provided type. Such an enum can be cast to a value of the same
type as well. In short, `#[repr(u8)]` makes the enum behave like an integer
with a constrained set of allowed values.

Only C-like enums can be cast to numerical primitives, so this attribute will
not apply to structs.

`#[repr(packed)]` reduces padding to make the struct size smaller. The
representation of enums isn't strictly defined in Rust, and this attribute
won't work on enums.

`#[repr(simd)]` will give a struct consisting of a homogenous series of machine
types (i.e. `u8`, `i32`, etc) a representation that permits vectorization via
SIMD. This doesn't make much sense for enums since they don't consist of a
single list of data.
"##,

E0518: r##"
This error indicates that an `#[inline(..)]` attribute was incorrectly placed
on something other than a function or method.

Examples of erroneous code:

```compile_fail,E0518
#[inline(always)]
struct Foo;

#[inline(never)]
impl Foo {
    // ...
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

E0522: r##"
The lang attribute is intended for marking special items that are built-in to
Rust itself. This includes special traits (like `Copy` and `Sized`) that affect
how the compiler behaves, as well as special functions that may be automatically
invoked (such as the handler for out-of-bounds accesses when indexing a slice).
Erroneous code example:

```compile_fail,E0522
#![feature(lang_items)]

#[lang = "cookie"]
fn cookie() -> ! { // error: definition of an unknown language item: `cookie`
    loop {}
}
```
"##,

E0525: r##"
A closure was used but didn't implement the expected trait.

Erroneous code example:

```compile_fail,E0525
struct X;

fn foo<T>(_: T) {}
fn bar<T: Fn(u32)>(_: T) {}

fn main() {
    let x = X;
    let closure = |_| foo(x); // error: expected a closure that implements
                              //        the `Fn` trait, but this closure only
                              //        implements `FnOnce`
    bar(closure);
}
```

In the example above, `closure` is an `FnOnce` closure whereas the `bar`
function expected an `Fn` closure. In this case, it's simple to fix the issue,
you just have to implement `Copy` and `Clone` traits on `struct X` and it'll
be ok:

```
#[derive(Clone, Copy)] // We implement `Clone` and `Copy` traits.
struct X;

fn foo<T>(_: T) {}
fn bar<T: Fn(u32)>(_: T) {}

fn main() {
    let x = X;
    let closure = |_| foo(x);
    bar(closure); // ok!
}
```

To understand better how closures work in Rust, read:
https://doc.rust-lang.org/book/closures.html
"##,

}


register_diagnostics! {
//  E0006 // merged with E0005
//  E0134,
//  E0135,
    E0278, // requirement is not satisfied
    E0279, // requirement is not satisfied
    E0280, // requirement is not satisfied
    E0284, // cannot resolve type
//  E0285, // overflow evaluation builtin bounds
//  E0300, // unexpanded macro
//  E0304, // expected signed integer constant
//  E0305, // expected constant
    E0311, // thing may not live long enough
    E0313, // lifetime of borrowed pointer outlives lifetime of captured variable
    E0314, // closure outlives stack frame
    E0315, // cannot invoke closure outside of its lifetime
    E0316, // nested quantification of lifetimes
    E0473, // dereference of reference outside its lifetime
    E0474, // captured variable `..` does not outlive the enclosing closure
    E0475, // index of slice outside its lifetime
    E0476, // lifetime of the source pointer does not outlive lifetime bound...
    E0477, // the type `..` does not fulfill the required lifetime...
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
    E0495, // cannot infer an appropriate lifetime due to conflicting requirements
    E0566  // conflicting representation hints
}
