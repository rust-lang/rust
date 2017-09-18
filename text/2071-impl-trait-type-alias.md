- Feature Name: impl-trait-existential-types
- Start Date: 2017-07-20
- RFC PR: https://github.com/rust-lang/rfcs/pull/2071
- Rust Issue: https://github.com/rust-lang/rust/issues/44685 (existential types)
- Rust Issue: https://github.com/rust-lang/rust/issues/44686 (impl Trait in const/static/let)

# Summary
[summary]: #summary

Add the ability to create named existential types and
support `impl Trait` in `let`, `const`, and `static` declarations.

```rust
// existential types
existential type Adder: Fn(usize) -> usize;
fn adder(a: usize) -> Adder {
    |b| a + b
}

// existential type in associated type position:
struct MyType;
impl Iterator for MyType {
    existential type Item: Debug;
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
it possible to create a named existential type:

```rust
// `impl Trait` in traits:
struct MyStruct;
impl Iterator for MyStruct {

    // Here we can declare an associated type whose concrete type is hidden
    // to other modules.
    //
    // External users only know that `Item` implements the `Debug` trait.
    existential type Item: Debug;

    fn next(&mut self) -> Option<Self::Item> {
        Some("hello")
    }
}
```

This syntax allows us to declare multiple items which refer to
the same existential type:

```rust
// Type `Foo` refers to a type that implements the `Debug` trait.
// The concrete type to which `Foo` refers is inferred from this module,
// and this concrete type is hidden from outer modules (but not submodules).
pub existential type Foo: Debug;

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

## Guide: Existential types
[guide-existential]: #guide-existential

Rust allows users to declare `existential type`s.
An existential type allows you to give a name to a type without revealing
exactly what type is being used.

```rust
use std::fmt::Debug;

existential type Foo: Debug;

fn foo() -> Foo {
    5i32
}
```

In the example above, `Foo` refers to `i32`, similar to a type alias.
However, unlike a normal type alias, the concrete type of `Foo` is
hidden outside of the module. Outside the module, the only think that
is known about `Foo` is that it implements the traits that appear in
its declaration (e.g. `Debug` in `existential type Foo: Debug;`).
If a user outside the module tries to use a `Foo` as an `i32`, they
will see an error:

```rust
use std::fmt::Debug;

mod my_mod {
  pub existential type Foo: Debug;

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
    // Creates a variable `x` of type `Foo`, which is only known to implement `Debug`
    let x = my_mod::foo();

    // Because we're outside `my_mod`, the user cannot determine the type of `Foo`.
    let y: i32 = my_mod::foo(); // ERROR: expected type `i32`, found existential type `Foo`

    // However, the user can use its `Debug` impl:
    println!("{:?}", x);
}
```

This makes it possible to write modules that hide their concrete types from the
outside world, allowing them to change implementation details without affecting
consumers of their API.

Note that it is sometimes necessary to manually specify the concrete type of an
existential type, like in `let x: i32 = foo();` above. This aids the function's
ability to locally infer the concrete type of `Foo`.

One particularly noteworthy use of existential types is in trait
implementations.
With this feature, we can declare associated types as follows:

```rust
struct MyType;
impl Iterator for MyType {
    existential type Item: Debug;
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
    existential type Item: Fn(i32) -> i32;
    fn next(&mut self) -> Option<Self::Item> {
        Some(|x| x + 5)
    }
}
```

Existential types can also be used to reference unnameable types in a struct
definition:

```rust
existential type Foo: Debug;
fn foo() -> Foo { 5i32 }

struct ContainsFoo {
    some_foo: Foo
}
```


It's also possible to write generic existential types:

```rust
#[derive(Debug)]
struct MyStruct<T: Debug> {
    inner: T
};

existential type Foo<T>: Debug;

fn get_foo<T: Debug>(x: T) -> Foo<T> {
    MyStruct {
        inner: x
    }
}
```

Similarly to `impl Trait` under
[RFC 1951](https://github.com/rust-lang/rfcs/blob/master/text/1951-expand-impl-trait.md),
`existential type` implicitly captures all generic type parameters in scope. In
practice, this means that existential associated types may contain generic
parameters from their impl:

```rust
struct MyStruct;
trait Foo<T> {
    type Bar;
    fn bar() -> Bar;
}

impl<T> Foo<T> for MyStruct {
    existentail type Bar: Trait;
    fn bar() -> Self::Bar {
        ...
        // Returns some type MyBar<T>
    }
}
```

However, as in 1951, lifetime parameters must be explicitly annotated.

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

## Reference: Existential Types
[reference-existential]: #reference-existential

Existential types are similar to normal type aliases, except that their
concrete type is determined from the scope in which they are defined
(usually a module or a trait impl).
For example, the following code has to examine the body of `foo` in order to
determine that the concrete type of `Foo` is `i32`:

```rust
existential type Foo = impl Debug;

fn foo() -> Foo {
    5i32
}
```

`Foo` can be used as `i32` in multiple places throughout the module.
However, each function that uses `Foo` as `i32` must independently place
constraints upon `Foo` such that it *must* be `i32`:

```rust
fn add_to_foo_1(x: Foo) {
    x + 1 // ERROR: binary operation `+` cannot be applied to existential type `Foo`
//  ^ `x` here is type `Foo`.
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

Each existential type declaration must be constrained by at least
one function body or const/static initializer.
A body or initializer must either fully constrain or place no constraints upon
a given existential type.

Outside of the module, existential types behave the same way as
`impl Trait` types: their concrete type is hidden from the module.
However, it can be assumed that two values of the same existential type
are actually values of the same type:

```rust
mod my_mod {
    pub existential type Foo: Debug;
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

One last difference between existential type aliases and normal type aliases is
that existential type aliases cannot be used in `impl` blocks:

```rust
existential type Foo: Debug;
impl Foo { // ERROR: `impl` cannot be used on existential type aliases
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
to express conditional bounds in existential type signatures
(e.g. `existentail type Foo<T> = impl Debug; impl<T: Clone> Clone for Foo<T> { ... }`).
This is a complicated design space which has not yet been explored fully
enough. In the future, such a feature could be added backwards-compatibly.

# Drawbacks
[drawbacks]: #drawbacks

This RFC proposes the addition of a complicated feature that will take time
for Rust developers to learn and understand.
There are potentially simpler ways to achieve some of the goals of this RFC,
such as making `impl Trait` usable in traits.
This RFC instead introduces a more complicated solution in order to
allow for increased expressiveness and clarity.

This RFC makes `impl Trait` feel even more like a type by allowing it in more
locations where formerly only concrete types were allowed.
However, there are other places such a type can appear where `impl Trait`
cannot, such as `impl` blocks and `struct` definitions
(i.e. `struct Foo { x: impl Trait }`).
This inconsistency may be surprising to users.

# Alternatives
[alternatives]: #alternatives

We could instead expand `impl Trait` in a more focused but limited way,
such as specifically extending `impl Trait` to work in traits without
allowing full existential type aliases.
A draft RFC for such a proposal can be seen
[here](https://github.com/cramertj/impl-trait-goals/blob/impl-trait-in-traits/0000-impl-trait-in-traits.md).
Any such feature could, in the future, be added as essentially syntax sugar on
top of this RFC, which is strictly more expressive.
The current RFC will also help us to gain experience with how people use
existential type aliases in practice, allowing us to resolve some remaining questions
in the linked draft, specifically around how `impl Trait` associated types
are used.

Throughout the process we have considered a number of alternative syntaxes for
existential types. The syntax `existential type Foo: Trait;` is intended to be
a placeholder for a more concise and accessible syntax, such as
`abstract type Foo: Trait;`. A variety of variations on this theme have been
considered:

- Instead of `abstract type`, it could be some single keyword like `abstype`.
- We could use a different keyword from `abstract`, like `opaque` or `exists`.
- We could omit a keyword altogether and use `type Foo: Trait;` syntax
(outside of trait definitions).

A more divergent alternative is not to have an "existentail type" feature at all,
but instead just have `impl Trait` be allowed in type alias position.
Everything written `existential type $NAME: $BOUND;` in this RFC would instead be
written `type $NAME = impl $BOUND;`.

This RFC opted to avoid the `type Foo = impl Trait;` syntax because of its
potential teaching difficulties.
As a result of [RFC 1951][1951], `impl Trait` is sometimes
universal quantiifcation and sometimes existential quantification. By providing
a separate syntax for "explicit" existential quantification, `impl Trait` can
be taught as a syntactic sugar for generics and existential types. By "just using
`impl Trait`" for named existential type declarations,
there would be no desugaring-based explanation for all forms of `impl Trait`.

This choice has some disadvantages in comparison impl Trait in type aliases:

- We introduce another new syntax on top of `impl Trait`, which inherently has
some costs.
- Users can't use it in a nested fashion without creating an addiitonal
existential type.

Because of these downsides, we are open to reconsidering this question with
more practical experience, and the final syntax is left as an unresolved
question for the RFC.

# Unresolved questions
[unresolved]: #unresolved-questions

As discussed in the [alternatives][alternatives] section above, we will need to
reconsider the optimal syntax before stabilizing this feature.

Additionally, the following extensions should be considered in the future:

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
