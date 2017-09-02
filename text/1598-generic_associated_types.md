- Feature Name: generic_associated_types
- Start Date: 2016-04-29
- RFC PR: [rust-lang/rfcs#1598](https://github.com/rust-lang/rfcs/pull/1598)
- Rust Issue: [rust-lang/rust#44265](https://github.com/rust-lang/rust/issues/44265)

# Summary
[summary]: #summary

Allow type constructors to be associated with traits. This is an incremental
step toward a more general feature commonly called "higher-kinded types," which
is often ranked highly as a requested feature by Rust users. This specific
feature (associated type constructors) resolves one of the most common use
cases for higher-kindedness, is a relatively simple extension to the type
system compared to other forms of higher-kinded polymorphism, and is forward
compatible with more complex forms of higher-kinded polymorphism that may be
introduced in the future.

# Motivation
[motivation]: #motivation

Consider the following trait as a representative motivating example:

```rust
trait StreamingIterator {
    type Item<'a>;
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
}
```

This trait is very useful - it allows for a kind of Iterator which yields
values which have a lifetime tied to the lifetime of the reference passed to
`next`. A particular obvious use case for this trait would be an iterator over
a vector which yields overlapping, mutable subslices with each iteration. Using
the standard `Iterator` interface, such an implementation would be invalid,
because each slice would be required to exist for as long as the iterator,
rather than for as long as the borrow initiated by `next`.

This trait cannot be expressed in Rust as it exists today, because it depends
on a sort of higher-kinded polymorphism. This RFC would extend Rust to include
that specific form of higher-kinded polymorphism, which is refered to here as
associated type constructors. This feature has a number of applications, but
the primary application is along the same lines as the `StreamingIterator`
trait: defining traits which yield types which have a lifetime tied to the
local borrowing of the receiver type.

# Detailed design
[design]: #detailed-design

## Background: What is kindedness?

"Higher-kinded types" is a vague term, conflating multiple language features
under a single banner, which can be inaccurate. As background, this RFC
includes a brief overview of the notion of kinds and kindedness. Kinds are
often called 'the type of a type,' the exact sort of unhelpful description that
only makes sense to someone who already understands what is being explained.
Instead, let's try to understand kinds by analogy to types.

In a well-typed language, every expression has a type. Many expressions have
what are sometimes called 'base types,' types which are primitive to the
language and which cannot be described in terms of other types. In Rust, the
types `bool`, `i64`, `usize`, and `char` are all prominent examples of base
types. In contrast, there are types which are formed by arranging other types -
functions are a good example of this. Consider this simple function:

```rust
fn not(x: bool) -> bool {
   !x
}
```

`not` has the type `bool -> bool` (my apologies for using a syntax different
from Rust's). Note that this is different from the type of `not(true)`, which
is `bool`. This difference is important to understanding higher-kindedness.

In the analysis of kinds, all of these types - `bool`, `char`, `bool -> bool`
and so on - have the kind `type`. Every type has the kind `type`. However,
`type` is a base kind, just as `bool` is a base type, and there are terms with
more complex kinds, such as `type -> type`. An example of a term of this kind
is `Vec`, which takes a type as a parameter and evaluates to a type. The
difference between the kind of `Vec` and the kind of `Vec<i32>` (which is
`type`) is analogous to the difference between the type of `not` and
`not(true)`. Note that `Vec<T>` has the kind `type`, just like `Vec<i32>`: even
though `T` is a type parameter, `Vec` is still being applied to a type, just
like `not(x)` still has the type `bool` even though `x` is a variable.

A relatively uncommon feature of Rust is that it has _two_ base kinds, whereas
many languages which deal with higher-kindedness only have the base kind
`type`. The other base kind of Rust is the lifetime parameter. If you have a
type like `Foo<'a>`, the kind of `Foo` is `lifetime -> type`.

Higher-kinded terms can take multiple arguments as well, of course. `Result`
has the kind `type, type -> type`. Given `vec::Iter<'a, T>` `vec::Iter` has the
kind `lifetime, type -> type`.

Terms of a higher kind are often called 'type operators'; the type operators
which evaluate to a type are called 'type constructors'. There are other type
operators which evaluate to other type operators, and there are even higher
order type operators, which take type operators as their argument (so they have
a kind like `(type -> type) -> type`). This RFC doesn't deal with anything as
exotic as that.

Specifically, the goal of this RFC is to allow type constructors to be
associated with traits, just as you can currently associate functions, types,
and consts with traits. There are other forms of polymorphism involving type
constructors, such as implementing traits for a type constructor instead of a
type, which are not a part of this RFC.

## Features of associated type constructors

### Declaring & assigning an associated type constructor

This RFC proposes a very simple syntax for defining an associated type
constructor, which looks a lot like the syntax for creating aliases for type
constructors. The goal of using this syntax is to avoid to creating roadblocks
for users who do not already understand higher kindedness.

```rust
trait StreamingIterator {
   type Item<'a>;
}
```

It is clear that the `Item` associated item is a type constructor, rather than
a type, because it has a type parameter attached to it.

Associated type constructors can be bounded, just like associated types can be:

```rust
trait Iterable {
    type Item<'a>;
    type Iter<'a>: Iterator<Item = Self::Item<'a>>;
    
    fn iter<'a>(&'a self) -> Self::Iter<'a>;
}
```

This bound is applied to the "output" of the type constructor, and the parameter
is treated as a higher rank parameter. That is, the above bound is roughly
equivalent to adding this bound to the trait:

```rust
for<'a> Self::Iter<'a>: Iterator<Item = Self::Item<'a>>
```

Assigning associated type constructors in impls is very similar to the syntax
for assigning associated types:

```rust
impl<T> StreamingIterator for StreamIterMut<T> {
    type Item<'a> = &'a mut [T];
    ...
}
```

### Using an associated type constructor to construct a type

Once a trait has an associated type constructor, it can be applied to any
parameters or concrete term that are in scope. This can be done both inside the
body of the trait and outside of it, using syntax which is analogous to the
syntax for using associated types. Here are some examples:

```rust
trait StreamingIterator {
    type Item<'a>;
    // Applying the lifetime parameter `'a` to `Self::Item` inside the trait.
    fn next<'a>(&'a self) -> Option<Self::Item<'a>>;
}

struct Foo<T: StreamingIterator> {
    // Applying a concrete lifetime to the constructor outside the trait.
    bar: <T as StreamingIterator>::Item<'static>;
}
```

Associated type constructors can also be used to construct other type
constructors:

```rust
trait Foo {
    type Bar<'a, 'b>;
}

trait Baz {
    type Quux<'a>;
}

impl<T> Baz for T where T: Foo {
    type Quux<'a> = <T as Foo>::Bar<'a, 'static>;
}
```

Lastly, lifetimes can be elided in associated type constructors in the same
manner that they can be elided in other type constructors. Considering lifetime
ellision, the full definition of `StreamingIterator` is:

```rust
trait StreamingIterator {
    type Item<'a>;
    fn next(&mut self) -> Option<Self::Item>;
}
```

### Using associated type constructors in bounds

Users can bound parameters by the type constructed by that trait's associated
type constructor of a trait using HRTB. Both type equality bounds and trait
bounds of this kind are valid:

```rust
fn foo<T: for<'a> StreamingIterator<Item<'a>=&'a [i32]>>(iter: T) { ... }

fn foo<T>(iter: T) where T: StreamingIterator, for<'a> T::Item<'a>: Display { ... }
```

This RFC does not propose allowing any sort of bound by the type constructor
itself, whether an equality bound or a trait bound (trait bounds of course are
also impossible). 

## Associated type constructors of type arguments

All of the examples in this RFC have focused on associated type constructors of
lifetime arguments, however, this RFC proposes adding ATCs of types as well:

```rust
trait Foo {
    type Bar<T>;
}
```

This RFC does **not** propose extending HRTBs to take type arguments, which
makes these less expressive than they could be. Such an extension is desired,
but out of scope for this RFC.

Type arguments can be used to encode other forms of higher kinded polymorphism
using the "family" pattern. For example, Using the `PointerFamily` trait, you
can abstract over Arc and Rc:

```rust
trait PointerFamily {
    type Pointer<T>: Deref<Target = T>;
    fn new<T>(value: T) -> Self::Pointer<T>;
}

struct ArcFamily;

impl PointerFamily for ArcFamily {
    type Pointer<T> = Arc<T>;
    fn new<T>(value: T) -> Self::Pointer<T> {
        Arc::new(value)
    }
}

struct RcFamily;

impl PointerFamily for RcFamily {
    type Pointer<T> = Rc<T>;
    fn new<T>(value: T) -> Self::Pointer<T> {
        Rc::new(value)
    }
}

struct Foo<P: PointerFamily> {
    bar: P::Pointer<String>,
}
```

## Evaluating bounds and where clauses

### Bounds on associated type constructors

Bounds on associated type constructors are treated as higher rank bounds on the
trait itself. This makes their behavior consistent with the behavior of bounds
on regular associated types. For example:

```rust
trait Foo {
    type Assoc<'a>: Trait<'a>;
}
```

Is equivalent to:

```rust
trait Foo where for<'a> Self::Assoc<'a>: Trait<'a> {
    type Assoc<'a>;
}
```

### `where` clauses on associated types

In contrast, where clauses on associated types introduce constraints which must
be proven each time the associated type is used. For example:

```rust
trait Foo {
    type Assoc where Self: Sized;
}
```

Each invokation of `<T as Foo>::Assoc` will need to prove `T: Sized`, as
opposed to the impl needing to prove the bound as in other cases.

(@nikomatsakis believes that where clauses will be needed on associated type
constructors specifically to handle lifetime well formedness in some cases.
The exact details are left out of this RFC because they will emerge more fully
during implementation.)

## Benefits of implementing only this feature before other higher-kinded polymorphisms

This feature is not full-blown higher-kinded polymorphism, and does not allow
for the forms of abstraction that are so popular in Haskell, but it does
provide most of the unique-to-Rust use cases for higher-kinded polymorphism,
such as streaming iterators and collection traits. It is probably also the
most accessible feature for most users, being somewhat easy to understand
intuitively without understanding higher-kindedness.

This feature has several tricky implementation challenges, but avoids all of
these features that other kinds of higher-kinded polymorphism require:

* Defining higher-kinded traits
* Implementing higher-kinded traits for type operators
* Higher order type operators
* Type operator parameters bound by higher-kinded traits
* Type operator parameters applied to a given type or type parameter

## Advantages of proposed syntax

The advantage of the proposed syntax is that it leverages syntax that already
exists. Type constructors can already be aliased in Rust using the same syntax
that this used, and while type aliases play no polymorphic role in type
resolution, to users they seem very similar to associated types. A goal of this
syntax is that many users will be able to use types which have assocaited type
constructors without even being aware that this has something to do with a type
system feature called higher-kindedness.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This RFC uses the terminology "associated type constructor," which has become
the standard way to talk about this feature in the Rust community. This is not
a very accessible framing of this concept; in particular the term "type
constructor" is an obscure piece of jargon from type theory which most users
cannot be expected to be familiar with.

Upon accepting this RFC, we should begin (with haste) refering to this concept
as simply "generic associated types." Today, associated types cannot be
generic; after this RFC, this will be possible. Rather than teaching this as
a separate feature, it will be taught as an advanced use case for associated
types.

Patterns like "family traits" should also be taught in some way, possible in
the book or possibly just through supplemental forms of documentation like
blog posts.

This will also likely increase the frequency with which users have to employ
higher rank trait bounds; we will want to put additional effort into teaching
and making teachable HRTBs.

# Drawbacks
[drawbacks]: #drawbacks

## Adding language complexity

This would add a somewhat complex feature to the language, being able to
polymorphically resolve type constructors, and requires several extensions to
the type system which make the implementation more complicated.

Additionally, though the syntax is designed to make this feature easy to learn,
it also makes it more plausible that a user may accidentally use it when they
mean something else, similar to the confusion between `impl .. for Trait` and
`impl<T> .. for T where T: Trait`. For example:

```rust
// The user means this
trait Foo<'a> {
    type Bar: 'a;
}

// But they write this
trait Foo<'a> {
    type Bar<'a>;
}
```

## Not full "higher-kinded types"

This does not add all of the features people want when they talk about higher-
kinded types. For example, it does not enable traits like `Monad`. Some people
may prefer to implement all of these features together at once. However, this
feature is forward compatible with other kinds of higher-kinded polymorphism,
and doesn't preclude implementing them in any way. In fact, it paves the way
by solving some implementation details that will impact other kinds of higher-
kindedness as well, such as partial application.

## Syntax isn't like other forms of higher-kinded polymorphism

Though the proposed syntax is very similar to the syntax for associated types
and type aliases, it is probably not possible for other forms of higher-kinded
polymorphism to use a syntax along the same lines. For this reason, the syntax
used to define an associated type constructor will probably be very different
from the syntax used to e.g. implement a trait for a type constructor.

However, the syntax used for these other forms of higher-kinded polymorphism
will depend on exactly what features they enable. It would be hard to design
a syntax which is consistent with unknown features.

# Alternatives
[alternatives]: #alternatives

## Push HRTBs harder without associated type constructors

An alternative is to push harder on HRTBs, possibly introducing some elision
that would make them easier to use.

Currently, an approximation of `StreamingIterator` can be defined like this:

```rust
trait StreamingIterator<'a> {
   type Item: 'a;
   fn next(&'a self) -> Option<Self::Item>;
}
```

You can then bound types as `T: for<'a> StreamingIterator<'a>` to avoid the
lifetime parameter infecting everything `StreamingIterator` appears.

However, this only partially prevents the infectiveness of `StreamingIterator`,
only allows for some of the types that associated type constructors can
express, and is in generally a hacky attempt to work around the limitation
rather than an equivalent alternative.

## Impose restrictions on ATCs

What is often called "full higher kinded polymorphism" is allowing the use of
type constructors as input parameters to other type constructors - higher order
type constructors, in other words. Without any restrictions, multiparameter
higher order type constructors present serious problems for type inference.

For example, if you are attempting to infer types, and you know you have a
constructor of the form `type, type -> Result<(), io::Error>`, without any
restrictions it is difficult to determine if this constructor is
`(), io::Error -> Result<(), io::Error>` or `io::Error, () -> Result<(), io::Error>`.

Because of this, languages with first class higher kinded polymorphism tend to
impose restrictions on these higher kinded terms, such as Haskell's currying
rules.

If Rust were to adopt higher order type constructors, it would need to impose
similar restrictions on the kinds of type constructors they can receive. But
associated type constructors, being a kind of alias, inherently mask the actual
structure of the concrete type constructor. In other words, if we want to be
able to use ATCs as arguments to higher order type constructors, we would need
to impose those restrictions on *all* ATCs.

We have a list of restrictions we believe are necessary and sufficient; more
background can be found in [this blog post](http://smallcultfollowing.com/babysteps/blog/2016/11/09/associated-type-constructors-part-4-unifying-atc-and-hkt/)
by nmatsakis:

* Each argument to the ATC must be applied
* They must be applied in the same order they appear in the ATC
* They must be applied exactly once
* They must be the left-most arguments of the constructor

These restrictions are quite constrictive; there are several applications of
ATCs that we already know about that would be frustrated by this, such as the
definition of `Iterable` for `HashMap` (for which the item `(&'a K, &'a V)`,
applying the lifetime twice).

For this reason we have decided **not** to apply these restrictions to all
ATCs. This will mean that if higher order type constructors are ever added to
the language, they will not be able to take an abstract ATC as an argument.
However, this can be maneuvered around using newtypes which do meet the
restrictions, for example:

```rust
struct IterItem<'a, I: Iterable>(I::Item<'a>);
```

# Unresolved questions
[unresolved]: #unresolved-questions
