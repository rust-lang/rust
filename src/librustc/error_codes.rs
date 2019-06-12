#![allow(non_snake_case)]

// Error messages for EXXXX errors.
// Each message should start and end with a new line, and be wrapped to 80 characters.
// In vim you can `:set tw=80` and use `gq` to wrap paragraphs. Use `:set tw=0` to disable.
register_long_diagnostics! {
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

Generally, `Self: Sized` is used to indicate that the trait should not be used
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

```compile_fail,E0038
# trait Trait { fn foo<T>(&self, on: T); }
# impl Trait for String { fn foo<T>(&self, on: T) {} }
# impl Trait for u8 { fn foo<T>(&self, on: T) {} }
# impl Trait for bool { fn foo<T>(&self, on: T) {} }
# // etc.
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
trait object (e.g., if `T: OtherTrait`, use `on: Box<OtherTrait>`). If the
number of types you intend to feed to this method is limited, consider manually
listing out the methods of different types.

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

### The trait cannot contain associated constants

Just like static functions, associated constants aren't stored on the method
table. If the trait or any subtrait contain an associated constant, they cannot
be made into an object.

```compile_fail,E0038
trait Foo {
    const X: i32;
}

impl Foo {}
```

A simple workaround is to use a helper method instead:

```
trait Foo {
    fn x(&self) -> i32;
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

E0106: r##"
This error indicates that a lifetime is missing from a type. If it is an error
inside a function signature, the problem may be with failing to adhere to the
lifetime elision rules (see below).

Here are some simple examples of where you'll run into this error:

```compile_fail,E0106
struct Foo1 { x: &bool }
              // ^ expected lifetime parameter
struct Foo2<'a> { x: &'a bool } // correct

struct Bar1 { x: Foo2 }
              // ^^^^ expected lifetime parameter
struct Bar2<'a> { x: Foo2<'a> } // correct

enum Baz1 { A(u8), B(&bool), }
                  // ^ expected lifetime parameter
enum Baz2<'a> { A(u8), B(&'a bool), } // correct

type MyStr1 = &str;
           // ^ expected lifetime parameter
type MyStr2<'a> = &'a str; // correct
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

[book-le]: https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#lifetime-elision
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

E0139: r##"
#### Note: this error code is no longer emitted by the compiler.

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

```
use std::mem::transmute;

struct Foo<T>(Vec<T>);

trait MyTransmutableType: Sized {
    fn transmute(_: Vec<Self>) -> Foo<Self>;
}

impl MyTransmutableType for u8 {
    fn transmute(x: Vec<u8>) -> Foo<u8> {
        unsafe { transmute(x) }
    }
}

impl MyTransmutableType for String {
    fn transmute(x: Vec<String>) -> Foo<String> {
        unsafe { transmute(x) }
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
# use std::ptr;
# let v = Some("value");
# type SomeType = &'static [u8];
unsafe {
    ptr::read(&v as *const _ as *const SomeType) // `v` transmuted to `SomeType`
}
# ;
```

Note that this does not move `v` (unlike `transmute`), and may need a
call to `mem::forget(v)` in case you want to avoid destructors being called.
"##,

E0152: r##"
A lang item was redefined.

Erroneous code example:

```compile_fail,E0152
#![feature(lang_items)]

#[lang = "arc"]
struct Foo; // error: duplicate lang item found: `arc`
```

Lang items are already implemented in the standard library. Unless you are
writing a free-standing application (e.g., a kernel), you do not need to provide
them yourself.

You can build a free-standing crate by adding `#![no_std]` to the crate
attributes:

```ignore (only-for-syntax-highlight)
#![no_std]
```

See also the [unstable book][1].

[1]: https://doc.rust-lang.org/unstable-book/language-features/lang-items.html#writing-an-executable-without-stdlib
"##,

E0214: r##"
A generic type was described using parentheses rather than angle brackets.
For example:

```compile_fail,E0214
fn main() {
    let v: Vec(&str) = vec!["foo"];
}
```

This is not currently supported: `v` should be defined as `Vec<&str>`.
Parentheses are currently only used with generic types when defining parameters
for `Fn`-family traits.
"##,

E0230: r##"
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
of the same type; e.g., a literal `{` is `{{`.
"##,

E0231: r##"
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

E0232: r##"
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
struct Foo<'a> {
    x: &'a str,
}

fn foo<'a>(x: &'a str) {}
```

Impl blocks declare lifetime parameters separately. You need to add lifetime
parameters to an impl block if you're implementing a type that has a lifetime
parameter of its own.
For example:

```compile_fail,E0261
struct Foo<'a> {
    x: &'a str,
}

// error,  use of undeclared lifetime name `'a`
impl Foo<'a> {
    fn foo<'a>(x: &'a str) {}
}
```

This is fixed by declaring the impl block like this:

```
struct Foo<'a> {
    x: &'a str,
}

// correct
impl<'a> Foo<'a> {
    fn foo(x: &'a str) {}
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
    #[lang = "panic_impl"] // ok!
    fn cake();
}
```
"##,

E0271: r##"
This is because of a type mismatch between the associated type of some
trait (e.g., `T::Bar`, where `T` implements `trait Quux { type Bar; }`)
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

```compile_fail,E0271
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
//~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#### Note: this error code is no longer emitted by the compiler.

You tried to supply a type which doesn't implement some trait in a location
which expected that trait. This error typically occurs when working with
`Fn`-based types. Erroneous code example:

```compile-fail
fn foo<F: Fn(usize)>(x: F) { }

fn main() {
    // type mismatch: ... implements the trait `core::ops::Fn<(String,)>`,
    // but the trait `core::ops::Fn<(usize,)>` is required
    // [E0281]
    foo(|y: String| { });
}
```

The issue in this case is that `foo` is defined as accepting a `Fn` with one
argument of type `String`, but the closure we attempted to pass to it requires
one arguments of type `usize`.
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
    // Should it be Impl or AnotherImpl, maybe something else?
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

E0284: r##"
This error occurs when the compiler is unable to unambiguously infer the
return type of a function or method which is generic on return type, such
as the `collect` method for `Iterator`s.

For example:

```compile_fail,E0284
fn foo() -> Result<bool, ()> {
    let results = [Ok(true), Ok(false), Err(())].iter().cloned();
    let v: Vec<bool> = results.collect()?;
    // Do things with v...
    Ok(true)
}
```

Here we have an iterator `results` over `Result<bool, ()>`.
Hence, `results.collect()` can return any type implementing
`FromIterator<Result<bool, ()>>`. On the other hand, the
`?` operator can accept any type implementing `Try`.

The author of this code probably wants `collect()` to return a
`Result<Vec<bool>, ()>`, but the compiler can't be sure
that there isn't another type `T` implementing both `Try` and
`FromIterator<Result<bool, ()>>` in scope such that
`T::Ok == Vec<bool>`. Hence, this code is ambiguous and an error
is returned.

To resolve this error, use a concrete type for the intermediate expression:

```
fn foo() -> Result<bool, ()> {
    let results = [Ok(true), Ok(false), Err(())].iter().cloned();
    let v = {
        let temp: Result<Vec<bool>, ()> = results.collect();
        temp?
    };
    // Do things with v...
    Ok(true)
}
```

Note that the type of `v` can now be inferred from the type of `temp`.
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
The type definition contains some field whose type
requires an outlives annotation. Outlives annotations
(e.g., `T: 'a`) are used to guarantee that all the data in T is valid
for at least the lifetime `'a`. This scenario most commonly
arises when the type contains an associated type reference
like `<T as SomeTrait<'a>>::Output`, as shown in this example:

```compile_fail,E0309
// This won't compile because the applicable impl of
// `SomeTrait` (below) requires that `T: 'a`, but the struct does
// not have a matching where-clause.
struct Foo<'a, T> {
    foo: <T as SomeTrait<'a>>::Output,
}

trait SomeTrait<'a> {
    type Output;
}

impl<'a, T> SomeTrait<'a> for T
where
    T: 'a,
{
    type Output = u32;
}
```

Here, the where clause `T: 'a` that appears on the impl is not known to be
satisfied on the struct. To make this example compile, you have to add
a where-clause like `T: 'a` to the struct definition:

```
struct Foo<'a, T>
where
    T: 'a,
{
    foo: <T as SomeTrait<'a>>::Output
}

trait SomeTrait<'a> {
    type Output;
}

impl<'a, T> SomeTrait<'a> for T
where
    T: 'a,
{
    type Output = u32;
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

E0398: r##"
#### Note: this error code is no longer emitted by the compiler.

In Rust 1.3, the default object lifetime bounds are expected to change, as
described in [RFC 1156]. You are getting a warning because the compiler
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

```
# trait SomeTrait {}
fn foo(arg: &Box<SomeTrait>) { /* ... */ }
```

You might change it to:

```
# trait SomeTrait {}
fn foo<'a>(arg: &'a Box<SomeTrait+'a>) { /* ... */ }
```

This explicitly states that you expect the trait object `SomeTrait` to contain
references (with a maximum lifetime of `'a`).

[RFC 1156]: https://github.com/rust-lang/rfcs/blob/master/text/1156-adjust-default-object-bounds.md
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
trait SomeTrait<'a> {
    type Output;
}

impl<'a, T> SomeTrait<'a> for T {
    type Output = &'a T; // compile error E0491
}
```

Here, the problem is that a reference type like `&'a T` is only valid
if all the data in T outlives the lifetime `'a`. But this impl as written
is applicable to any lifetime `'a` and any type `T` -- we have no guarantee
that `T` outlives `'a`. To fix this, you can add a where clause like
`where T: 'a`.

```
trait SomeTrait<'a> {
    type Output;
}

impl<'a, T> SomeTrait<'a> for T
where
    T: 'a,
{
    type Output = &'a T; // compile error E0491
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
discriminant size for enums with no data fields on any of the variants, e.g.
`enum Color {Red, Blue, Green}`, effectively setting the size of the enum to
the size of the provided type. Such an enum can be cast to a value of the same
type as well. In short, `#[repr(u8)]` makes the enum behave like an integer
with a constrained set of allowed values.

Only field-less enums can be cast to numerical primitives, so this attribute
will not apply to structs.

`#[repr(packed)]` reduces padding to make the struct size smaller. The
representation of enums isn't strictly defined in Rust, and this attribute
won't work on enums.

`#[repr(simd)]` will give a struct consisting of a homogeneous series of machine
types (i.e., `u8`, `i32`, etc) a representation that permits vectorization via
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
https://doc.rust-lang.org/book/ch13-01-closures.html
"##,

E0580: r##"
The `main` function was incorrectly declared.

Erroneous code example:

```compile_fail,E0580
fn main(x: i32) { // error: main function has wrong type
    println!("{}", x);
}
```

The `main` function prototype should never take arguments.
Example:

```
fn main() {
    // your code
}
```

If you want to get command-line arguments, use `std::env::args`. To exit with a
specified exit code, use `std::process::exit`.
"##,

E0562: r##"
Abstract return types (written `impl Trait` for some trait `Trait`) are only
allowed as function and inherent impl return types.

Erroneous code example:

```compile_fail,E0562
fn main() {
    let count_to_ten: impl Iterator<Item=usize> = 0..10;
    // error: `impl Trait` not allowed outside of function and inherent method
    //        return types
    for i in count_to_ten {
        println!("{}", i);
    }
}
```

Make sure `impl Trait` only appears in return-type position.

```
fn count_to_n(n: usize) -> impl Iterator<Item=usize> {
    0..n
}

fn main() {
    for i in count_to_n(10) {  // ok!
        println!("{}", i);
    }
}
```

See [RFC 1522] for more details.

[RFC 1522]: https://github.com/rust-lang/rfcs/blob/master/text/1522-conservative-impl-trait.md
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

E0593: r##"
You tried to supply an `Fn`-based type with an incorrect number of arguments
than what was expected.

Erroneous code example:

```compile_fail,E0593
fn foo<F: Fn()>(x: F) { }

fn main() {
    // [E0593] closure takes 1 argument but 0 arguments are required
    foo(|y| { });
}
```
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

E0602: r##"
An unknown lint was used on the command line.

Erroneous example:

```sh
rustc -D bogus omse_file.rs
```

Maybe you just misspelled the lint name or the lint doesn't exist anymore.
Either way, try to update/remove it in order to fix the error.
"##,

E0621: r##"
This error code indicates a mismatch between the lifetimes appearing in the
function signature (i.e., the parameter types and the return type) and the
data-flow found in the function body.

Erroneous code example:

```compile_fail,E0621
fn foo<'a>(x: &'a i32, y: &i32) -> &'a i32 { // error: explicit lifetime
                                             //        required in the type of
                                             //        `y`
    if x > y { x } else { y }
}
```

In the code above, the function is returning data borrowed from either `x` or
`y`, but the `'a` annotation indicates that it is returning data only from `x`.
To fix the error, the signature and the body must be made to match. Typically,
this is done by updating the function signature. So, in this case, we change
the type of `y` to `&'a i32`, like so:

```
fn foo<'a>(x: &'a i32, y: &'a i32) -> &'a i32 {
    if x > y { x } else { y }
}
```

Now the signature indicates that the function data borrowed from either `x` or
`y`. Alternatively, you could change the body to not return data from `y`:

```
fn foo<'a>(x: &'a i32, y: &i32) -> &'a i32 {
    x
}
```
"##,

E0635: r##"
The `#![feature]` attribute specified an unknown feature.

Erroneous code example:

```compile_fail,E0635
#![feature(nonexistent_rust_feature)] // error: unknown feature
```

"##,

E0636: r##"
A `#![feature]` attribute was declared multiple times.

Erroneous code example:

```compile_fail,E0636
#![allow(stable_features)]
#![feature(rust1)]
#![feature(rust1)] // error: the feature `rust1` has already been declared
```

"##,

E0644: r##"
A closure or generator was constructed that references its own type.

Erroneous example:

```compile-fail,E0644
fn fix<F>(f: &F)
  where F: Fn(&F)
{
  f(&f);
}

fn main() {
  fix(&|y| {
    // Here, when `x` is called, the parameter `y` is equal to `x`.
  });
}
```

Rust does not permit a closure to directly reference its own type,
either through an argument (as in the example above) or by capturing
itself through its environment. This restriction helps keep closure
inference tractable.

The easiest fix is to rewrite your closure into a top-level function,
or into a method. In some cases, you may also be able to have your
closure call itself by capturing a `&Fn()` object or `fn()` pointer
that refers to itself. That is permitting, since the closure would be
invoking itself via a virtual call, and hence does not directly
reference its own *type*.

"##,

E0692: r##"
A `repr(transparent)` type was also annotated with other, incompatible
representation hints.

Erroneous code example:

```compile_fail,E0692
#[repr(transparent, C)] // error: incompatible representation hints
struct Grams(f32);
```

A type annotated as `repr(transparent)` delegates all representation concerns to
another type, so adding more representation hints is contradictory. Remove
either the `transparent` hint or the other hints, like this:

```
#[repr(transparent)]
struct Grams(f32);
```

Alternatively, move the other attributes to the contained type:

```
#[repr(C)]
struct Foo {
    x: i32,
    // ...
}

#[repr(transparent)]
struct FooWrapper(Foo);
```

Note that introducing another `struct` just to have a place for the other
attributes may have unintended side effects on the representation:

```
#[repr(transparent)]
struct Grams(f32);

#[repr(C)]
struct Float(f32);

#[repr(transparent)]
struct Grams2(Float); // this is not equivalent to `Grams` above
```

Here, `Grams2` is a not equivalent to `Grams` -- the former transparently wraps
a (non-transparent) struct containing a single float, while `Grams` is a
transparent wrapper around a float. This can make a difference for the ABI.
"##,

E0698: r##"
When using generators (or async) all type variables must be bound so a
generator can be constructed.

Erroneous code example:

```edition2018,compile-fail,E0698
#![feature(futures_api, async_await, await_macro)]
async fn bar<T>() -> () {}

async fn foo() {
  await!(bar());  // error: cannot infer type for `T`
}
```

In the above example `T` is unknowable by the compiler.
To fix this you must bind `T` to a concrete type such as `String`
so that a generator can then be constructed:

```edition2018
#![feature(futures_api, async_await, await_macro)]
async fn bar<T>() -> () {}

async fn foo() {
  await!(bar::<String>());
  //          ^^^^^^^^ specify type explicitly
}
```
"##,

E0700: r##"
The `impl Trait` return type captures lifetime parameters that do not
appear within the `impl Trait` itself.

Erroneous code example:

```compile-fail,E0700
use std::cell::Cell;

trait Trait<'a> { }

impl<'a, 'b> Trait<'b> for Cell<&'a u32> { }

fn foo<'x, 'y>(x: Cell<&'x u32>) -> impl Trait<'y>
where 'x: 'y
{
    x
}
```

Here, the function `foo` returns a value of type `Cell<&'x u32>`,
which references the lifetime `'x`. However, the return type is
declared as `impl Trait<'y>` -- this indicates that `foo` returns
"some type that implements `Trait<'y>`", but it also indicates that
the return type **only captures data referencing the lifetime `'y`**.
In this case, though, we are referencing data with lifetime `'x`, so
this function is in error.

To fix this, you must reference the lifetime `'x` from the return
type. For example, changing the return type to `impl Trait<'y> + 'x`
would work:

```
use std::cell::Cell;

trait Trait<'a> { }

impl<'a,'b> Trait<'b> for Cell<&'a u32> { }

fn foo<'x, 'y>(x: Cell<&'x u32>) -> impl Trait<'y> + 'x
where 'x: 'y
{
    x
}
```
"##,

E0701: r##"
This error indicates that a `#[non_exhaustive]` attribute was incorrectly placed
on something other than a struct or enum.

Examples of erroneous code:

```compile_fail,E0701
# #![feature(non_exhaustive)]

#[non_exhaustive]
trait Foo { }
```
"##,

E0718: r##"
This error indicates that a `#[lang = ".."]` attribute was placed
on the wrong type of item.

Examples of erroneous code:

```compile_fail,E0718
#![feature(lang_items)]

#[lang = "arc"]
static X: u32 = 42;
```
"##,

}


register_diagnostics! {
//  E0006, // merged with E0005
//  E0101, // replaced with E0282
//  E0102, // replaced with E0282
//  E0134,
//  E0135,
//  E0272, // on_unimplemented #0
//  E0273, // on_unimplemented #1
//  E0274, // on_unimplemented #2
    E0278, // requirement is not satisfied
    E0279, // requirement is not satisfied
    E0280, // requirement is not satisfied
//  E0285, // overflow evaluation builtin bounds
//  E0296, // replaced with a generic attribute input check
//  E0300, // unexpanded macro
//  E0304, // expected signed integer constant
//  E0305, // expected constant
    E0311, // thing may not live long enough
    E0312, // lifetime of reference outlives lifetime of borrowed content
    E0313, // lifetime of borrowed pointer outlives lifetime of captured variable
    E0314, // closure outlives stack frame
    E0315, // cannot invoke closure outside of its lifetime
    E0316, // nested quantification of lifetimes
    E0320, // recursive overflow during dropck
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
    E0566, // conflicting representation hints
    E0623, // lifetime mismatch where both parameters are anonymous regions
    E0628, // generators cannot have explicit arguments
    E0631, // type mismatch in closure arguments
    E0637, // "'_" is not a valid lifetime bound
    E0657, // `impl Trait` can only capture lifetimes bound at the fn level
    E0687, // in-band lifetimes cannot be used in `fn`/`Fn` syntax
    E0688, // in-band lifetimes cannot be mixed with explicit lifetime binders
    E0697, // closures cannot be static
    E0707, // multiple elided lifetimes used in arguments of `async fn`
    E0708, // `async` non-`move` closures with arguments are not currently supported
    E0709, // multiple different lifetimes used in arguments of `async fn`
    E0710, // an unknown tool name found in scoped lint
    E0711, // a feature has been declared with conflicting stability attributes
//  E0702, // replaced with a generic attribute input check
    E0726, // non-explicit (not `'_`) elided lifetime in unsupported position
    E0727, // `async` generators are not yet supported
    E0728, // `await` must be in an `async` function or block
}
