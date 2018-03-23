- Feature Name: `self_in_typedefs`
- Start Date: 2018-01-17
- RFC PR: [rust-lang/rfcs#2300](https://github.com/rust-lang/rfcs/pull/2300)
- Rust Issue: [rust-lang/rust#49303](https://github.com/rust-lang/rust/issues/49303)

# Summary
[summary]: #summary

The special `Self` identifier is now permitted in `struct`, `enum`, and `union`
type definitions. A simple example `struct` is:

```rust
enum List<T>
where
    Self: PartialOrd<Self> // <-- Notice the `Self` instead of `List<T>`
{
    Nil,
    Cons(T, Box<Self>) // <-- And here.
}
```

# Motivation
[motivation]: #motivation

## Removing exceptions and making the language more uniform

The contextual identifier `Self` can already be used in type context in cases
such as when defining what an associated type is for a particular type as well
as for generic parameters in `impl`s as in:

```rust
trait Foo<T = Self> {
    type Bar;

    fn wibble<U>() where Self: Sized;
}

struct Quux;

impl Foo<Self> for Quux {
    type Bar = Self;

    fn wibble<U>() where Self: Sized {}
}
```

But this is not currently possible inside both fields and where clauses of
type definitions. This makes the language less consistent with respect to what
is allowed in type positions than what it could be.

## Principle of least surprise

Users, just new to the language and experts in the language alike, also
have a reasonable expectations that using `Self` inside type definitions is
in fact already possible. Users may have and have these expectations because
`Self` already works in other places where a type is expected. If a user
attempts to use `Self` today, that attempt will fail, breaking the users
intuition of the languages semantics. Avoiding that breakage will reduce the
paper cuts newcomers face when using the language. It will also allow the
community to focus on answering more important questions.

## Better ergonomics with smaller edit distances

When you have complex recursive `enum`s with many variants and generic types,
and want to rename a type parameter or the type itself, it would make renaming
and refactoring the type definitions easier if you did not have to make changes
in the variant fields which mention the type. This can be helped by IDEs to some
extent, but you do not always have such IDEs and even then, the readability of
using `Self` is superior to repeating the type in variants and fields since it
is a more visual cue that can be highlighted for specially.

## Encouraging descriptively named types, type variables, and more generic code

Making it simpler and more ergonomic to have longer type names and more
generic parameters in type definitions can also encourage using more
descriptive identifiers for both the type and the type variables used.
It may also encourage more generic code altogether.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

[An Obligatory Public Service Announcement]: http://cglab.ca/~abeinges/blah/too-many-lists/book/#an-obligatory-public-service-announcement

> [An Obligatory Public Service Announcement]: When reading this RFC,
> keep in mind that these lists are only examples.
> **Always consider if you really need to use linked lists!**

We will now go through a few examples of what you can and can't do with this RFC.

## Simple example

Let's look at a simple cons-list of `u8`s. Before this RFC, you had to write:

```rust
enum U8List {
    Nil,
    Cons(u8, Box<U8List>)
}
```

But with this RFC, you can now instead write:

```rust
enum U8List {
    Nil,
    Cons(u8, Box<Self>) // <-- Notice 'Self' here
}
```

If you had written this example with `Self` without this RFC,
the compiler would have greeted you with:

```
error[E0411]: cannot find type `Self` in this scope
 --> src/main.rs:3:18
  |
3 |     Cons(u8, Box<Self>) // <-- Notice 'Self' here
  |                  ^^^^ `Self` is only available in traits and impls
```

With this RFC, the compiler will never do so.

This new way of writing with `Self` can be thought of as literally
desugaring to the way it is written in the example before it. This also
extends to generic types (non-nullary type constructors) that are recursive.

## With generic type parameters

Continuing with the cons lists, let's take a look at how the canonical
linked-list example can be rewritten using this RFC.

We start off with:

```rust
enum List<T> {
    Nil,
    Cons(T, Box<List<T>>)
}
```

With this RFC, the snippet above can be rewritten as:

```rust
enum List<T> {
    Nil,
    Cons(T, Box<Self>) // <-- Notice 'Self' here
}
```

Notice in particular how we used just `Self` for both `U8List` and `List<T>`.
This applies to types with any number of parameters, including those that are
parameterized by lifetimes.

## Examples with lifetimes

An example of this can be seen in the following cons list:

```rust
enum StackList<'a, T: 'a> {
    Nil,
    Cons(T, &'a StackList<'a, T>)
}
```

which is rewritten with this RFC as:

```rust
enum StackList<'a, T: 'a> {
    Nil,
    Cons(T, &'a Self) // <-- Still using just 'Self'
}
```

## Structs and unions

You can also use `Self` in `struct`s as in:

```rust
struct NonEmptyList<T> {
    head: T,
    tail: Option<Box<NonEmptyList<T>>>,
}
```

which is written with this RFC as:

```rust
struct NonEmptyList<T> {
    head: T,
    tail: Option<Box<Self>>,
}
```

This also extends to `union`s.

## `where`-clauses

In today's Rust, it is possible to define a type such as:

```rust
struct Foo<T>
where
    Foo<T>: SomeTrait
{
    // Some fields..
}
```

and with some `impl`s:

```rust
trait SomeTrait { ... }

impl SomeTrait for Foo<u32> { ... }
impl SomeTrait for Foo<String> { ... }
```

this idiom bounds the types that the type parameter `T` can be of but also
avoids defining an `Auxiliary` trait which one bound `T` with as in:

```rust
struct Foo<T: Auxiliary> {
    // Some fields..
}
```

You could also have the type on the right hand side of the bound in the `where`
clause as in:

```rust
struct Bar<T>
where
    T: PartialEq<Bar<T>>
{
    // Some fields..
}
```

with this RFC, you can now redefine `Foo<T>` and `Bar<T>` as:

```rust
struct Foo<T>
where
    Self: SomeTrait // <-- Notice `Self`!
{
    // Some fields..
}

struct Bar<T>
where
    T: PartialEq<Self> // <-- Notice `Self`!
{
    // Some fields..
}
```

This makes the bound involving `Self` slightly more clear.

## When `Self` can **not** be used

Consider the following small expression language:

```rust
trait Ty { type Repr: ::std::fmt::Debug; }

#[derive(Debug)]
struct Int;
impl Ty for Int { type Repr = usize; }

#[derive(Debug)]
struct Bool;
impl Ty for Bool { type Repr = bool; }

#[derive(Debug)]
enum Expr<T: Ty> {
    Lit(T::Repr),
    Add(Box<Expr<Int>>, Box<Expr<Int>>),
    If(Box<Expr<Bool>>, Box<Expr<T>>, Box<Expr<T>>),
}

fn main() {
    let expr: Expr<Int> =
        Expr::If(
            Box::new(Expr::Lit(true)),
            Box::new(Expr::Lit(1)),
            Box::new(Expr::Add(
                Box::new(Expr::Lit(1)),
                Box::new(Expr::Lit(1))
            ))
        );
    println!("{:#?}", expr);
}
```

You may perhaps reach for this:

```rust
#[derive(Debug)]
enum Expr<T: Ty> {
    Lit(T::Repr),
    Add(Box<Self>, Box<Self>),
    If(Box<Self>, Box<Self>, Box<Self>),
}
```

But you have now changed the definition of `Expr` semantically.
The changed semantics are due to the fact that `Self` in this context is not
the same type as `Expr<Int>` or `Expr<Bool>`. The compiler, when desugaring
`Self` in this context, will simply substitute `Self` with what it sees in
`Expr<T: Ty>` (with any bounds removed).

You may at most use `Self` by changing the definition of `Expr<T>` to:

```rust
#[derive(Debug)]
enum Expr<T: Ty> {
    Lit(T::Repr),
    Add(Box<Expr<Int>>, Box<Expr<Int>>),
    If(Box<Expr<Bool>>, Box<Self>, Box<Self>),
}
```

## Types of infinite size

Consider the following example:

```rust
enum List<T> {
    Nil,
    Cons(T, List<T>)
}
```

If you try to compile it this today, the compiler will greet you with:

```
error[E0072]: recursive type `List` has infinite size
 --> src/main.rs:1:1
  |
1 | enum List<T> {
  | ^^^^^^^^^^^^ recursive type has infinite size
2 |     Nil,
3 |     Cons(T, List<T>)
  |             -------- recursive without indirection
  |
  = help: insert indirection (e.g., a `Box`, `Rc`, or `&`) at some point to make `List` representable
```

If we use the syntax introduced by this RFC as in:

```rust
enum List<T> {
    Nil,
    Cons(T, Self)
}
```

you will still get an error since
[it is fundamentally impossible to construct a type of infinite size][E0072].
The error message would however use `Self` as you wrote it instead of `List<T>`
as seen in this snippet:

[E0072]: https://doc.rust-lang.org/error-index.html#E0072

```
error[E0072]: recursive type `List` has infinite size
 --> src/main.rs:1:1
  |
1 | enum List<T> {
  | ^^^^^^^^^^^^ recursive type has infinite size
2 |     Nil,
3 |     Cons(T, Self)
  |             ----- recursive without indirection
  |
  = help: insert indirection (e.g., a `Box`, `Rc`, or `&`) at some point to make `List` representable
```

## Teaching the contents of this RFC

[LRWETMLL]: http://cglab.ca/~abeinges/blah/too-many-lists/book/first-layout.html

When talking about and teaching recursive types in Rust, since it is now
possible to use `Self`, the ability to use `Self` in this context should
be taught along side those types. An example of where this can be introduced
is the [*"Learning Rust With Entirely Too Many Linked Lists"* guide][LRWETMLL].

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The identifier `Self` is (now) allowed in type contexts in fields of `struct`s,
`union`s, and the variants of `enum`s. The identifier `Self` is also allowed
as the left hand side of a bound in a `where` clause and as a type argument
to a trait bound on the right hand side of a `where` clause.

## Desugaring

When the compiler encounters `Self` in type contexts inside the places
described above, it will substitute them with the header of the type
definition but remove any bounds on generic parameters prior.

An example: the following cons list:

```rust
enum StackList<'a, T: 'a + InterestingTrait> {
    Nil,
    Cons(T, &'a Self)
}
```

desugars into:

```rust
enum StackList<'a, T: 'a + InterestingTrait> {
    Nil,
    Cons(T, &'a StackList<'a, T>)
}
```

Note in particular that the source code is **not** desugared into:

```rust
enum StackList<'a, T: 'a + InterestingTrait> {
    Nil,
    Cons(T, &'a StackList<'a, T: 'a + InterestingTrait>)
}
```

An example of `Self` in `where` bounds is:

```rust
struct Foo<T>
where
    Self: PartialEq<Self>
{
    // Some fields..
}
```

which desugars into:

```rust
struct Foo<T>
where
    Foo<T>: PartialEq<Foo<T>>
{
    // Some fields..
}
```

[RFC 2102]: https://github.com/rust-lang/rfcs/pull/2102

## In relation to [RFC 2102] and what `Self` refers to.

It should be noted that `Self` always refers to the top level type and not
the inner unnamed `struct` or `union` because those are unnamed. Specifically,
*Self always applies to the innermost nameable type*. In type definitions in
particular, this is equivalent: *Self always applies to the top level type*.

## Error messages

When `Self` is used to construct an infinite type as in:

```rust
enum List<T> {
    Nil,
    Cons(T, Self)
}
```

The compiler will emit error `E0072` as in:

```
error[E0072]: recursive type `List` has infinite size
 --> src/main.rs:1:1
  |
1 | enum List<T> {
  | ^^^^^^^^^^^^ recursive type has infinite size
2 |     Nil,
3 |     Cons(T, Self)
  |             ----- recursive without indirection
  |
  = help: insert indirection (e.g., a `Box`, `Rc`, or `&`) at some point to make `List` representable
```

Note in particular that `Self` is used and not `List<T>` on line `3`.

## In relation to other RFCs

This RFC expands on [RFC 593] and [RFC 1647] with respect to where the keyword
`Self` is allowed.

[RFC 593]: 0593-forbid-Self-definitions.md
[RFC 1647]: 1647-allow-self-in-where-clauses.md

# Drawbacks
[drawbacks]: #drawbacks

Some may argue that we shouldn't have many ways to do the same thing and
that it introduces new syntax whereby making the surface language more complex.
However, the RFC may equally be said to simplify the surface language since
it removes exceptional cases especially in the users mental model.

Using `Self` in a type definition makes it harder to search for all positions
in which a pattern can appear in an AST.

# Rationale and alternatives
[alternatives]: #alternatives

The rationale for this particular design is straightforward as it would be
uneconomic, confusing, and inconsistent to use other keywords.

## The consistency of what `Self` refers to

As explained in the [reference-level explanation], we said that:
> *Self always applies to the innermost nameable type*.

We arrive at this conclusion by examining a few different cases and what
they have in common.

### Current Rust - Shadowing in `impl`s

First, let's take a look at shadowing in `impl`s.

```rust
fn main() { Foo {}.foo(); }

#[derive(Debug)]
struct Foo;

impl Foo {
    fn foo(&self) {
        // Prints "Foo", which is the innermost type.
        println!("{:?}", Self {});

        #[derive(Debug)]
        struct Bar;

        impl Bar {
            fn bar(&self) {
                // Prints "Bar", also the innermost type in this context.
                println!("{:?}", Self {});
            }
        }
        Bar {}.bar();
    }
}
```

Let's also consider trait impls instead of inherent impls:

```rust
impl Trait for Foo {
    fn foo(&self) {
        impl Trait for Bar {
            // Self is shadowed here...
        }
    }
}
```

We see that the conclusion holds for both examples.

### In relation to [RFC 2102]

Let's consider a modified example from [RFC 2102]:

```rust
#[repr(C)]
struct S {
    a: u32,
    _: union {
        b: Box<Self>,
        c: f32,
    },
    d: u64,
}
```

In this example, the inner union is not nameable, and so `Self` refers to the
only nameable introduced type `S`. Therefore, the conclusion holds.

### Type definitions inside `impl`s

If in the future we decide to permit type definitions inside `impl`s as in:

```rust
impl Trait for Foo {
    struct Bar {
        head: u8,
        tail: Option<Box<Self>>,
    }
}
```

as sugar for:

```rust
enum _Bar {
    head: u8,
    tail: Option<Box<Self>>,
}
impl Trait for Foo {
    type Bar = _Bar;
}
```

In the desugared example, we see that the only possible meaning of `Self` is
that it refers to `_Bar` and not `Foo`. To be consistent with the desugared
form, the sugared variant should have the same meaning and so `Self` refers
to `Bar` there.

Let's now consider an alternative possible syntax:

```rust
impl Trait for Foo {
    type Bar = struct /* there is no ident here */ {
        outer: Option<Box<Self>>,
        inner: Option<Box<Self::Item>>,
    }
}
```

Notice here in particular that there is no identifier after the keyword
`struct`. Because of this, it is reasonable to say that the `struct`
assigned to the associated type `Bar` is not directly nameable as `Bar`.
Instead, a user must qualify `Bar` with `Self::Bar`. With this in mind,
we arrive at the following interpretation:

```rust
impl Trait for Foo {
    type Bar = struct /* there is no ident here */ {
        outer: Option<Box<Foo>>,
        inner: Option<Box<Foo::Bar>>,
    }
}
```

### Conclusion

We've now examined a few cases and seen that indeed, the meaning of `Self` is
consistent in all of them as well as with what the meaning in in today's Rust.

## Doing nothing

One alternative to the changes proposed in this RFC is to simply not implement
those changes. However, this has the downsides of not increasing the ergonomics
and keeping the language less consistent than what it could be. Not improving
the ergonomics here may be especially problematic when dealing with "recursive"
types that have long names and/or many generic parameters and may encourage
developers to use type names which are less descriptive and keep their code
less generic than what is appropriate.

## Internal scoped type aliases

Another alternative is to allow users to specify type aliases inside type
definitions and use any generic parameters specified in that definition.
An example is:

```rust
enum Tree<T> {
    type S = Box<Tree<T>>;

    Nil,
    Node(T, S, S),
}
```

instead of:

```rust
enum Tree<T> {
    Nil,
    Node(T, Box<Self>, Box<Self>),
}
```

[generic associated types]: https://github.com/rust-lang/rfcs/blob/master/text/1598-generic_associated_types.md

When dealing with *[generic associated types] (GATs)*, we can then write:

```rust
enum Tree<T, P: PointerFamily> {
    type S = P::Pointer<Tree<T>>;

    Nil,
    Node(T, S, S),
}
```

instead of:

```rust
enum Tree<T, P: PointerFamily> {
    Nil,
    Node(T, P::Pointer<Tree<T>>, P::Pointer<Tree<T>>),
}
```

As we can see, this approach and alternative is more flexible compared to
what is proposed in this RFC, particularly in the case of GATs. However,
this alternative requires introducing and teaching more concepts compared
to this RFC, which comparatively builds more on what users already know.
Mixing `;` and `,` has also proven to be controversial in the past. The
alternative also opens up questions such as if the type alias should be
permitted before the variants, or after the variants.

For simpler cases such as the first tree-example, using `Self` is also more
readable as it is a special construct that you can easily syntax-highlight
for in a more noticeable way. Further, while there is an expectation from
some users that `Self` already works, as discussed in the [motivation],
the expectation that this alternative already works has not been brought
forth by anyone as far as this RFC's author is aware.

It is also unclear how internal scoped type aliases would syntactically work
with `where` bounds.

Strictly speaking, this particular alternative is not in conflict with this
RFC in that both can be supported technically. The alternative should be
considered interesting future work, but for now, a more conservative approach
is preferred.

# Unresolved questions
[unresolved]: #unresolved-questions

+ This syntax creates ambiguity if we ever permit types to be declared directly
within impls (for example, as the value for an associated type). Do we ever want
to support that, and if so, how should we resolve the ambiguity? **A** possible,
interpretation and way to solve the ambiguity consistently is discussed in the
rationale.
