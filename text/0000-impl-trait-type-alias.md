- Feature Name: impl-trait-type-alias
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add the ability to create type aliases for `impl Trait` types,
and support `impl Trait` in `let`, `const`, and `static` declarations.

```rust
// `impl Trait` type alias:
type Adder = impl Fn(usize) -> usize;
fn adder(a: usize) -> Adder {
    |b| a + b
}

// `impl Trait` type aliases in associated type position:
struct MyType;
impl Iterator for MyType {
    type Item = impl Debug;
    fn next(&mut self) -> Option<Self::Item> {
        Some("Another item!")
    }
}

// `impl Trait` in `let`, `const`, and `static`:

const ADD_ONE: impl Fn(usize) -> usize = |x| x + 1;

static MAYBE_PRINT: Option<impl Fn(usize)> = Some(|x| println!("{}", x));

fn my_func() {
    let iter: impl Iterator<Item = i32> = (0..5).map(|x| x * 5);
    ...
}
```

# Motivation
[motivation]: #motivation

This RFC proposes two expansions to Rust's `impl Trait` feature.
`impl Trait`, first introduced in [RFC 1522][1522], allows functions to return
types which implement a given trait, but whose concrete type remains anonymous.
`impl Trait` was expanded upon in [RFC 1951][1951], which added `impl Trait` to
argument position and resolved questions around syntax and parameter scoping.
In its current form, the feature makes it possible for functions to return
unnameable or complex types such as closures and iterator combinators.
`impl Trait` also allows library authors to hide the concrete type returned by
a function, making it possible to change the return type later on.

However, the current feature has some severe limitations.
Right now, it isn't possible to return an `impl Trait` type from a trait
implementation. This is a huge restriction which this RFC fixes by making
it possible to create a type alias for an `impl Trait` type using the syntax
`type Foo = impl Trait;`. The type alias syntax also makes it possible to
guarantee that two `impl Trait` types are the same:

```rust
// `impl Trait` in traits:
struct MyStruct;
impl Iterator for MyStruct {

    // Here we can declare an associated type whose concrete type is hidden
    // to other modules.
    //
    // External users only know that `Item` implements the `Debug` trait.
    type Item = impl Debug;

    fn next(&mut self) -> Option<Self::Item> {
        Some("hello")
    }
}
```

`impl Trait` type aliases allow us to declare multiple items which refer to
the same `impl Trait` type:

```rust
// Type `Foo` refers to a type that implements the `Debug` trait.
// The concrete type to which `Foo` refers is inferred from this module,
// and this concrete type is hidden from outer modules (but not submodules).
pub type Foo = impl Debug;

const FOO: Foo = 5;

// This function can be used by outer modules to manufacture an instance of
// `Foo`. Other modules don't know the concrete type of `Foo`,
// so they can't make their own `Foo`s.
pub fn get_foo() -> Foo {
    5
}

// We know that the argument and return value of `get_larger_foo` must be the
// same type as is returned from `get_foo`.
pub fn get_larger_foo(x: Foo) -> Foo {
    let x: i32 = x;
    x + 10
}

// Since we know that all `Foo`s have the same (hidden) concrete type, we can
// write a function which returns `Foo`s acquired from different places.
fn one_of_the_foos(which: usize) -> Foo {
    match which {
        0 => FOO,
        1 => foo1(),
        2 => foo2(),
        3 => opt_foo().unwrap(),

        // It also allows us to make recursive calls to functions with an
        // `impl Trait` return type:
        x => one_of_the_foos(x - 4),
    }
}
```

Separately, this RFC adds the ability to store an `impl Trait` type in a
`let`, `const` or `static`.
This makes `const` and `static` declarations more concise,
and makes it possible to store types such as closures or iterator combinators
in `const`s and `static`s.

In a future world where `const fn` has been expanded to trait functions,
one could imagine iterator constants such as this:

```rust
const THREES: impl Iterator<Item = i32> = (0..).map(|x| x * 3);
```

Since the type of `THREES` contains a closure, it is impossible to write down.
The [`const`/`static` type annotation elison RFC][2010] has suggested one
possible solution.
That RFC proposes to let users omit the types of `const`s and `statics`s.
However, in some cases, completely omitting the types of `const` and `static`
items could make it harder to tell what sort of value is being stored in a
`const` or `static`.
Allowing `impl Trait` in `const`s and `static`s would resolve the unnameable
type issue while still allowing users to provide some information about the
type.

[1522]: https://github.com/rust-lang/rfcs/blob/master/text/1522-conservative-impl-trait.md
[1951]: https://github.com/rust-lang/rfcs/blob/master/text/1951-expand-impl-trait.md
[2010]: https://github.com/rust-lang/rfcs/pull/2010

# Guide-Level Explanation
[guide]: #guide

## Guide: `impl Trait` in `let`, `const` and `static`:
[guide-declarations]: #guide-declarations

`impl Trait` can be used in `let`, `const`, and `static` declarations,
like this:
```rust
use std::fmt::Display;

let displayable: impl Display = "Hello, world!";
println!("{}", displayable);
```

Declaring a variable of type `impl Trait` will hide its concrete type.
This is useful for declaring a value which implements a trait,
but whose concrete type might change later on.
In our example above, this means that, while we can "display" the
value of `displayable`, the concrete type `&str` is hidden:

```rust
use std::fmt::Display;

// Without `impl Trait`:
const DISPLAYABLE: &str = "Hello, world!";
fn display() {
    println!("{}", DISPLAYABLE);
    assert_eq!(DISPLAYABLE.len(), 5);
}

// With `impl Trait`:
const DISPLAYABLE: impl Display = "Hello, world!";

fn display() {
    // We know `DISPLAYABLE` implements `Display`.
    println!("{}", DISPLAYABLE);

    // ERROR: no method `len` on `impl Display`
    // We don't know the concrete type of `DISPLAYABLE`,
    // so we don't know that it has a `len` method.
    assert_eq!(DISPLAYABLE.len(), 5);
}
```

`impl Trait` declarations are also useful when declaring constants or
static with types that are impossible to name, like closures:

```rust
// Without `impl Trait`, we can't declare this constant because we can't
// write down the type of the closure.
const MY_CLOSURE: ??? = |x| x + 1;

// With `impl Trait`:
const MY_CLOSURE: impl Fn(i32) -> i32 = |x| x + 1;
```

Finally, note that `impl Trait` `let` declarations hide the concrete
types of local variables:

```rust
let displayable: impl Display = "Hello, world!";

// We know `displayable` implements `Display`.
println!("{}", displayable);

// ERROR: no method `len` on `impl Display`
// We don't know the concrete type of `displayable`,
// so we don't know that it has a `len` method.
assert_eq!(displayable.len(), 5);
```

At first glance, this behavior doesn't seem particularly useful.
Indeed, `impl Trait` in `let` bindings exists mostly for consistency with
`const`s and `static`s. However, it can be useful for documenting the
specific ways in which a variable is used. It can also be used to provide
better error messages for complex, nested types:

```rust
// Without `impl Trait`:
let x = (0..100).map(|x| x * 3).filter(|x| x % 5 == 0);

// ERROR: no method named `bogus_missing_method` found for type
// `std::iter::Filter<std::iter::Map<std::ops::Range<{integer}>, [closure@src/main.rs:2:26: 2:35]>, [closure@src/main.rs:2:44: 2:58]>` in the current scope
x.bogus_missing_method();

// With `impl Trait`:
let x: impl Iterator<Item = i32> = (0..100).map(|x| x * 3).filter(|x| x % 5);

// ERROR: no method named `bogus_missing_method` found for type
// `impl std::iter::Iterator` in the current scope
x.bogus_missing_method();
```

## Guide: `impl Trait` Type Aliases
[guide-aliases]: #guide-aliases

`impl Trait` can also be used to create type aliases:

```rust
use std::fmt::Debug;

type Foo = impl Debug;

fn foo() -> Foo {
    5i32
}
```

`impl Trait` type aliases, just like regular type aliases, create
synonyms for a type.
In the example above, `Foo` is a synonym for `i32`.
The difference between `impl Trait` type aliases and regular type aliases is
that `impl Trait` type aliases hide their concrete type from other modules
(but not submodules).
Only the `impl Trait` signature is exposed:

```rust
use std::fmt::Debug;

mod my_mod {
  pub type Foo = impl Debug;

  pub fn foo() -> Foo {
      5i32
  }

  pub fn use_foo_inside_mod() -> Foo {
      // Creates a variable `x` of type `i32`, which is equal to type `Foo`
      let x: i32 = foo();
      x + 5
  }
}

fn use_foo_outside_mod() {
    // Creates a variable `x` of type `Foo`, which is equal to type `impl Debug`
    let x = my_mod::foo();

    // Because we're outside `my_mod`, using a value of type `Foo` as anything
    // other than `impl Debug` is an error:
    let y: i32 = my_mod::foo(); // ERROR: expected type `i32`, found type `Foo`
}
```

This makes it possible to write modules that hide their concrete types from the
outside world, allowing them to change implementation details without affecting
consumers of their API.

Note that it is sometimes necessary to manually specify the concrete type of an
`impl Trait`-aliased value, like in `let x: i32 = foo();` above.
This aids the function's ability to locally infer the concrete type of `Foo`.

One particularly noteworthy use of `impl Trait` type aliases is in trait
implementations.
With this feature, we can declare `impl Trait` associated types:

```rust
struct MyType;
impl Iterator for MyType {
    type Item = impl Debug;
    fn next(&mut self) -> Option<Self::Item> {
        Some("Another item!")
    }
}
```

In this trait implementation, we've declared that the item returned by our
iterator implements `Debug`, but we've kept its concrete type (`&'static str`)
hidden from the outside world.

We can even use this feature to specify unnameable associated types, such as
closures:

```rust
struct MyType;
impl Iterator for MyType {
    type Item = impl Fn(i32) -> i32;
    fn next(&mut self) -> Option<Self::Item> {
        Some(|x| x + 5)
    }
}
```

`impl Trait` aliases can also be used to reference unnameable types in a struct
definition:

```rust
type Foo = impl Debug;
fn foo() -> Foo { 5i32 }

struct ContainsFoo {
    some_foo: Foo
}
```


`impl Trait` can also appear inside of another type in a type alias:

```rust
type Foo = Option<impl Debug>;
fn foo() -> Foo {
    Some("Debuggable")
}
```

Or even multiple times within the same type alias:

```rust
type Foo = (impl Debug, impl Fn());
fn foo() -> Foo {
    ("Debuggable", || println!("Hello, world!"))
}
```

It's also possible to write generic `impl Trait` aliases:

```rust
#[derive(Debug)]
struct MyStruct<T: Debug> {
    inner: T
};

type Foo<T> = impl Debug;

fn get_foo<T: Debug>(x: T) -> Foo<T> {
    MyStruct {
        inner: x
    }
}
```

As specified in
[RFC 1951](https://github.com/rust-lang/rfcs/blob/master/text/1951-expand-impl-trait.md),
`impl Trait` implicitly captures all generic types parameters in scope.
In practice, this means that `impl Trait` associated types may contain generic
types from an impl:

```rust
struct MyStruct;
trait Foo<T> {
    type Bar;
    fn bar() -> Bar;
}

impl<T> Foo<T> for MyStruct {
    type Bar = impl Trait;
    fn bar() -> impl Trait {
        ...
        // Returns some type MyBar<T>
    }
}
```

However, as in 1951, `impl Trait` lifetime parameters must be explicitly
annotated.

# Reference-Level Explanation
[reference]: #reference

## Reference: `impl Trait` in `let`, `const` and `static`:
[reference-declarations]: #reference-declarations

The rules for `impl Trait` values in `let`, `const`, and `static` declarations
work mostly the same as `impl Trait` return values as specified in
[RFC 1951](https://github.com/rust-lang/rfcs/blob/master/text/1951-expand-impl-trait.md).

These values hide their concrete type and can only be used as a value which
is known to implement the specified traits. They inherit any type parameters
in scope. One difference from `impl Trait` return types is that they also
inherit any lifetime parameters in scope. This is necessary in order for
`let` bindings to use `impl Trait`. `let` bindings often contain references
which last for anonymous scope-based lifetimes, and annotating these lifetimes
manually would be impossible.

## Reference: `impl Trait` Type Aliases
[reference-aliases]: #reference-aliases

`impl Trait` type aliases are similar to normal type aliases, except that their
concrete type is determined from the scope in which they are defined
(usually a module or a trait impl).
For example, the following code has to examine the body of `foo` in order to
determine that the concrete type of `Foo` is `i32`:

```rust
type Foo = impl Debug;

fn foo() -> Foo {
    5i32
}
```

`Foo` can be used as `i32` in multiple places throughout the module.
However, each function that uses `Foo` as `i32` must independently place
constraints upon `Foo` such that it *must* be `i32`:

```rust
fn add_to_foo_1(x: Foo) {
    x + 1 // ERROR: binary operation `+` cannot be applied to type `impl Debug`
//  ^ `x` here is type `impl Debug`.
//    Type annotations needed to resolve the concrete type of `x`.
//    (^ This particular error should only appear within the module in which
//      `Foo` is defined)
}

fn add_to_foo_2(x: Foo) {
    let x: i32 = x;
    x + 1
}

fn return_foo(x: Foo) -> Foo {
    // This is allowed.
    // We don't need to know the concrete type of `Foo` for this function to
    // typecheck.
    x
}
```

Each instance of `impl Trait` in a type alias must be constrained by at least
one function body or const/static initializer.
A body or initializer must either fully constrain or place no constraints upon
a given instance of `impl Trait` in a type alias.

The following is an example of an `impl Trait` type alias which contains
two instances of `impl Trait`. Each instance is determined by exactly one
of the functions:

```rust
// The concrete type of `Baz` resolves to `(i32, &'static str)`
type Baz = (impl Default + Debug, impl Default + Debug);

// This function places no constraints on the `impl Trait` types
fn new_baz() -> Baz {
    (Default::default(), Default::default())
}

// This function fully constraints the first `impl Trait` type,
// but places no constraints upon the second.
fn add_to_first(baz: Baz) -> Baz {
    let first: i32 = baz.0;
    (first + 1, baz.1)
}

// This function fully constrains the second `impl Trait` type,
// but places no constraints upon the first.
fn make_second_hello(baz: Baz) -> Baz {
    (baz.0, "Hello, world!")
}
```

Outside of the module, `impl Trait` alias types behave the same way as
other `impl Trait` types, except that it can be assumed that two values with
the same `impl Trait` alias type are actually values of the same type:

```rust
mod my_mod {
    pub type Foo = impl Debug;
    pub fn foo() -> Foo {
        5i32
    }
    pub fn bar() -> Foo {
        10i32
    }
    pub fn baz(x: Foo) -> Foo {
        let x: i32 = x;
        x + 5
    }
}

fn outside_mod() -> Foo {
    if true {
        my_mod::foo()
    } else {
        my_mod::baz(my_mod::bar())
    }
}
```

One last difference between `impl Trait` aliases and normal type aliases is
that `impl Trait` aliases cannot be used in `impl` blocks:

```rust
type Foo = impl Debug;
impl Foo { // ERROR: `impl` cannot be used on `impl Trait` aliases
    ...
}
impl MyTrait for Foo { // ERROR ^
    ...
}
```

While this feature may be added at some point in the future, it's unclear
exactly what behavior it should have-- should it result in implementations
of functions and traits on the underlying type? It seems like the answer
should be "no" since doing so would give away the underlying type being
hidden beneath the impl. Still, some version of this feature could be
used eventually to implement traits or functions for closures, or
to express conditional bounds in `impl Trait` signatures
(e.g. `type Foo<T> = impl Debug; impl<T: Clone> Clone for Foo<T> { ... }`).
This is a complicated design space which has not yet been explored fully
enough. In the future, such a feature could be added backwards-compatibly.

# Drawbacks
[drawbacks]: #drawbacks

This RFC proposes the addition of a complicated feature that will take time
for Rust developers to learn and understand.
There are potentially simpler ways to acheive some of the goals of this RFC,
such as making `impl Trait` usable in traits.
This RFC instead introduces a more complicated solution in order to
allow for increased expressiveness and clarity.

This RFC makes `impl Trait` feel even more like a type by allowing it in more
locations where formerly only concrete types were allowed.
However, there are other places such a type can appear where `impl Trait`
cannot, such as `impl` blocks and `struct` definitions
(i.e. `struct Foo { x: impl Trait }`).
This inconsistency may be surprising to users.

Additionally, if Rust ever moves to a bare `Trait` (no `impl`) syntax,
`type Foo = impl Trait;` would likely require a new syntax, as
`type Foo = MyType;` and `type Foo = Trait;` have different-enough
behavior that the syntactic similarity would cause confusion.

# Alternatives
[alternatives]: #alternatives

We could instead expand `impl Trait` in a more focused but limited way,
such as specifically extending `impl Trait` to work in traits without
allowing full "`impl Trait` type alias".
A draft RFC for such a proposal can be seen
[here](https://github.com/cramertj/impl-trait-goals/blob/impl-trait-in-traits/0000-impl-trait-in-traits.md).
Any such feature could, in the future, be added as essentially syntax sugar on
top of this RFC, which is strictly more expressive.
The current RFC will also help us to gain experience with how people use
`impl Trait` in practice, allowing us to resolve some remaining questions
in the linked draft, specifically around how `impl Trait` associated types
are used.

There are a number of alternative syntaxes we could use for `impl Trait`
aliases:
- `abstype / abstract type Foo: Trait;`: Suggested in
[RFC 1951](https://github.com/rust-lang/rfcs/blob/master/text/1951-expand-impl-trait.md),
this syntax has the potential advantage of being able to specify constraints
such as `abstract type Foo: Trait = MyType;`, which provides module-level
abstraction without relying upon module-level inference.
- `type Foo: Trait;`: This option also has the "abstraction without inference"
advantage. However, it doesn't include an easily searchable keyword like
`abstract/abstype/impl`, so it might be hard for users to discover what's going
on when they first encounter this syntax.
- `type Foo = impl Trait;`: This is the syntax option I've used in this RFC.
It is the only option which doesn't allow for "abstraction without inference",
but it is also the only option which allows for "composite" `impl Trait` types
such as `type Foo = (impl Debug, impl Fn());`. It also bears a syntactic
resemblance to the `impl Trait` feature, which should make it easy for new users
to identify and understand.

# Unresolved questions
[unresolved]: #unresolved-questions

The following extensions should be considered in the future:

- Conditional bounds. Even with this proposal, there's no way to specify
the `impl Trait` bounds necessary to implement traits like `Iterator`, which
have functions whose return types implement traits conditional on the input,
e.g. `fn foo<T>(x: T) -> impl Clone if T: Clone`.
- Associated-type-less `impl Trait` in trait declarations and implementations,
such as the proposal mentioned in the alternatives section.
As mentioned above, this feature would be strictly less expressive than this
RFC. The more general feature proposed in this RFC would help us to define a
better version of this alternative which could be added in the future.
- A more general form of inference for `impl Trait` type aliases. This RFC
forces each function to either fully constrain or place no constraints upon
an `impl Trait` type. It's possible to allow some partial constraints through
a process like the one described in
[this comment](https://github.com/rust-lang/rfcs/pull/2071#issuecomment-320458113).
However, these partial bounds present implementation concerns, so they have
been removed from this RFC. If it turns out that partial bounds would be
greatly useful in practice, they can be added backwards-compatibly in a future
RFC.
