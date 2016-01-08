- Feature Name: specialization
- Start Date: 2015-06-17
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

This RFC proposes a design for *specialization*, which permits multiple `impl`
blocks to apply to the same type/trait, so long as there is always a clearly
"most specific" impl block that applies. The most specific `impl` block is used
in a case of overlap. The design proposed here also supports refining default
trait implementations based on specifics about the types involved.

Altogether, this relatively small extension to the trait system yields benefits
for performance, expressiveness, and code reuse, and it lays the groundwork for an "efficient
inheritance" scheme that is largely based on the trait system (described in a
forthcoming companion RFC).

# Motivation

Specialization brings benefits along several different axes:

* **Performance**: specialization expands the scope of "zero cost abstraction",
  because specialized impls can provide custom high-performance code for
  particular, concrete cases of an abstraction.

* **Expressiveness**: specialization significantly relaxes the overlapping impl
  rules, making it possible to write multiple blanket impls that intersect -- a
  desire that has come up over and over again in the standard library and
  elsewhere.

* **Reuse**: the design proposed here also supports refining default (but
  incomplete) implementations of a trait, given details about the types
  involved.

* **Groundwork**: the design lays the groundwork for supporting
  ["efficient inheritance"](https://internals.rust-lang.org/t/summary-of-efficient-inheritance-rfcs/494)
  through the trait system.

The following subsections dive into each of these motivations in more detail.

## Performance

The simplest and most longstanding motivation for specialization is
performance.

To take a very simple example, suppose we add a trait for overloading the `+=`
operator:

```rust
trait AddAssign<Rhs=Self> {
    fn add_assign(&mut self, Rhs);
}
```

It's tempting to provide an impl for any type that you can both `Clone` and
`Add`:

```rust
impl<R, T: Add<R> + Clone> AddAssign<R> for T {
    fn add_assign(&mut self, rhs: R) {
        let tmp = self.clone() + rhs;
        *self = tmp;
    }
}
```

This impl is especially nice because it means that you frequently don't have to
bound separately by `Add` and `AddAssign`; often `Add` is enough to give you
both operators.

However, in today's Rust, such an impl would rule out any more specialized
implementation that, for example, avoids the call to `clone`. That means there's
a tension between simple abstractions and code reuse on the one hand, and
performance on the other. Specialization resolves this tension by allowing both
the blanket impl, and more specific ones, to coexist, using the specialized ones
whenever possible (and thereby guaranteeing maximal performance).

More broadly, traits today can provide static dispatch in Rust, but they can
still impose an abstraction tax. For example, consider the `Extend` trait:

```rust
pub trait Extend<A> {
    fn extend<T>(&mut self, iterable: T) where T: IntoIterator<Item=A>;
}
```

Collections that implement the trait are able to insert data from arbitrary
iterators. Today, that means that the implementation can assume nothing about
the argument `iterable` that it's given except that it can be transformed into
an iterator. That means the code must work by repeatedly calling `next` and
inserting elements one at a time.

But in specific cases, like extending a vector with a slice, a much more
efficient implementation is possible -- and the optimizer isn't always capable
of producing it automatically. In such cases, specialization can be used to get
the best of both worlds: retaining the abstraction of `extend` while providing
custom code for specific cases.

The design in this RFC relies on multiple, overlapping trait impls, so to take
advantage for `Extend` we need to refactor a bit:

```rust
pub trait Extend<A, T: IntoIterator<Item=A>> {
    fn extend(&mut self, iterable: T);
}

// The generic implementation
impl<A, T> Extend<A, T> for Vec<A> where T: IntoIterator<Item=A> {
    // the `default` qualifier allows this method to be specialized below
    default fn extend(&mut self, iterable: T) {
        ... // implementation using push (like today's extend)
    }
}

// A specialized implementation for slices
impl<'a, A> Extend<A, &'a [A]> for Vec<A> {
    fn extend(&mut self, iterable: &'a [A]) {
        ... // implementation using ptr::write (like push_all)
    }
}
```

Other kinds of specialization are possible, including using marker traits like:

```rust
unsafe trait TrustedSizeHint {}
```

that can allow the optimization to apply to a broader set of types than slices,
but are still more specific than `T: IntoIterator`.

## Expressiveness

One frequent motivation for specialization is broader "expressiveness", in
particular providing a larger set of trait implementations than is possible
today.

For example, the standard library currently includes an `AsRef` trait
for "as-style" conversions:

```rust
pub trait AsRef<T> where T: ?Sized {
    fn as_ref(&self) -> &T;
}
```

Currently, there is also a blanket implementation as follows:

```rust
impl<'a, T: ?Sized, U: ?Sized> AsRef<U> for &'a T where T: AsRef<U> {
    fn as_ref(&self) -> &U {
        <T as AsRef<U>>::as_ref(*self)
    }
}
```

which allows these conversions to "lift" over references, which is in turn
important for making a number of standard library APIs ergonomic.

On the other hand, we'd also like to provide the following very simple
blanket implementation:

```rust
impl<'a, T: ?Sized> AsRef<T> for T {
    fn as_ref(&self) -> &T {
        self
    }
}
```

The current coherence rules prevent having both impls, however,
because they can in principle overlap:

```rust
AsRef<&'a T> for &'a T where T: AsRef<&'a T>
```

Another example comes from the `Option` type, which currently provides two
methods for unwrapping while providing a default value for the `None` case:

```rust
impl<T> Option<T> {
    fn unwrap_or(self, def: T) -> T { ... }
    fn unwrap_or_else<F>(self, f: F) -> T where F: FnOnce() -> T { .. }
}
```

The `unwrap_or` method is more ergonomic but `unwrap_or_else` is more efficient
in the case that the default is expensive to compute. The original
[collections reform RFC](https://github.com/rust-lang/rfcs/pull/235) proposed a
`ByNeed` trait that was rendered unworkable after unboxed closures landed:

```rust
trait ByNeed<T> {
    fn compute(self) -> T;
}

impl<T> ByNeed<T> for T {
    fn compute(self) -> T {
        self
    }
}

impl<F, T> ByNeed<T> for F where F: FnOnce() -> T {
    fn compute(self) -> T {
        self()
    }
}

impl<T> Option<T> {
    fn unwrap_or<U>(self, def: U) where U: ByNeed<T> { ... }
    ...
}
```

The trait represents any value that can produce a `T` on demand. But the above
impls fail to compile in today's Rust, because they overlap: consider `ByNeed<F>
for F` where `F: FnOnce() -> F`.

There are also some trait hierarchies where a subtrait completely subsumes the
functionality of a supertrait. For example, consider `PartialOrd` and `Ord`:

```rust
trait PartialOrd<Rhs: ?Sized = Self>: PartialEq<Rhs> {
    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;
}

trait Ord: Eq + PartialOrd<Self> {
    fn cmp(&self, other: &Self) -> Ordering;
}
```

In cases like this, it's somewhat annoying to have to provide an impl for *both*
`Ord` and `PartialOrd`, since the latter can be trivially derived from the
former. So you might want an impl like this:

```rust
impl<T> PartialOrd<T> for T where T: Ord {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
```

But this blanket impl would conflict with a number of others that work to "lift"
`PartialOrd` and `Ord` impls over various type constructors like references and
tuples, e.g.:

```rust
impl<'a, A: ?Sized> Ord for &'a A where A: Ord {
    fn cmp(&self, other: & &'a A) -> Ordering { Ord::cmp(*self, *other) }
}

impl<'a, 'b, A: ?Sized, B: ?Sized> PartialOrd<&'b B> for &'a A where A: PartialOrd<B> {
    fn partial_cmp(&self, other: &&'b B) -> Option<Ordering> {
        PartialOrd::partial_cmp(*self, *other)
    }
```

The case where they overlap boils down to:

```rust
PartialOrd<&'a T> for &'a T where &'a T: Ord
PartialOrd<&'a T> for &'a T where T: PartialOrd
```

There are many other examples along these lines.

Specialization as proposed in this RFC greatly relaxes the overlap rules, even
allowing impls to only partially overlap, as long as there is another
yet-more-specialized impl that disambiguates the portion of partial overlap.

## Reuse

Today's default methods in traits are pretty limited: they can assume only the
`where` clauses provided by the trait itself, and there is no way to provide
conditional or refined defaults that rely on more specific type information.

For example, consider a different design for overloading `+` and `+=`, such that
they are always overloaded together:

```rust
trait Add<Rhs=Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
    fn add_assign(&mut self, Rhs);
}
```

In this case, there's no natural way to provide a default implementation of
`add_assign`, since we do not want to restrict the `Add` trait to `Clone` data.

The specialization design in this RFC also allows for *partial* implementations,
which can provide specialized defaults without actually providing a full trait
implementation:

```rust
partial impl<T: Clone, Rhs> Add<Rhs> for T {
    // the `default` qualifier allows further specialization
    default fn add_assign(&mut self, rhs: R) {
        let tmp = self.clone() + rhs;
        *self = tmp;
    }
}
```

This partial impl does *not* mean that `Add` is implemented for all `Clone`
data, but jut that when you do impl `Add` and `Self: Clone`, you can leave off
`add_assign`:

```rust
#[derive(Copy, Clone)]
struct Complex {
    // ...
}

impl Add<Complex> for Complex {
    type Output = Complex;
    fn add(self, rhs: Complex) {
        // ...
    }
    // no fn add_assign necessary
}
```

A particularly nice case of refined defaults comes from trait hierarchies: you
can sometimes use methods from subtraits to improve default supertrait
methods. For example, consider the relationship between `size_hint` and
`ExactSizeIterator`:

```rust
partial impl<T> Iterator for T where T: ExactSizeIterator {
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
```

As we'll see later, the design of this RFC makes it possible to "lock down" such
method impls (by not using the `default` qualifier), preventing any further
refinement (akin to Java's `final` keyword); that in turn makes it possible to
statically enforce the contract that is supposed to connect the `len` and
`size_hint` methods. (Of course, we can't make *that* particular change, since
the relevant APIs are already stable.)

## Supporting efficient inheritance

Finally, specialization can be seen as a form of inheritance, since methods
defined within a blanket impl can be overridden in a fine-grained way by a more
specialized impl. As we will see, this analogy is a useful guide to the design
of specialization. But it is more than that: the specialization design proposed
here is specifically tailored to support "efficient inheritance" schemes (like
those discussed
[here](https://internals.rust-lang.org/t/summary-of-efficient-inheritance-rfcs/494))
without adding an entirely separate inheritance mechanism.

The key insight supporting this design is that virtual method definitions in
languages like C++ and Java actually encompass two distinct mechanisms: virtual
dispatch (also known as "late binding") and implementation inheritance. These
two mechanisms can be separated and addressed independently; this RFC
encompasses an "implementation inheritance" mechanism distinct from virtual
dispatch, and useful in a number of other circumstances. But it can be combined
nicely with an orthogonal mechanism for virtual dispatch to give a complete
story for the "efficient inheritance" goal that many previous RFCs targeted.

The author is preparing a companion RFC showing how this can be done with a
relatively small further extension to the language. In the meantime, you can
find a blog post laying out the basic ideas
[here](http://aturon.github.io/blog/2015/09/18/reuse/).

But it should be said that the design in *this* RFC is fully motivated
independently of its companion RFC.

# Detailed design

There's a fair amount of material to cover, so we'll start with a basic overview
of the design in intuitive terms, and then look more formally at a specification.

At the simplest level, specialization is about allowing overlap between impl
blocks, so long as there is always an unambiguous "winner" for any type falling
into the overlap. For example:

```rust
impl<T> Debug for T where T: Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl Debug for String {
    fn fmt(&self, f: &mut Formatter) -> Result {
        try!(write!(f, "\""));
        for c in self.chars().flat_map(|c| c.escape_default()) {
            try!(write!(f, "{}", c));
        }
        write!(f, "\"")
    }
}
```

The idea for this pair of impls is that you can rest assured that *any* type
implementing `Display` will also implement `Debug` via a reasonable default, but
go on to provide more specific `Debug` implementations when warranted. In
particular, the intuition is that a `Self` type of `String` is somehow "more
specific" or "more concrete" than `T where T: Display`.

The bulk of the detailed design is aimed at making this intuition more
precise. But first, we need to explore some problems that arise when you
introduce specialization in any form.

## Hazard: interactions with type checking

Consider the following, somewhat odd example of overlapping impls:

```rust
trait Example {
    type Output;
    fn generate(self) -> Self::Output;
}

impl<T> Example for T {
    type Output = Box<T>;
    fn generate(self) -> Box<T> { Box::new(self) }
}

impl Example for bool {
    type Output = bool;
    fn generate(self) -> bool { self }
}
```

The key point to pay attention to here is the difference in associated types:
the blanket impl uses `Box<T>`, while the impl for `bool` just uses `bool`.
If we write some code that uses the above impls, we can get into trouble:

```rust
fn trouble<T>(t: T) -> Box<T> {
    Example::generate(t)
}

fn weaponize() -> bool {
    let b: Box<bool> = trouble(true);
    *b
}
```

What's going on? When type checking `trouble`, the compiler has a type `T` about
which it knows nothing, and sees an attempt to employ the `Example` trait via
`Example::generate(t)`. Because of the blanket impl, this use of `Example` is
allowed -- but furthermore, the associated type found in the blanket impl is now
directly usable, so that `<T as Example>::Output` is known within `trouble` to
be `Box<T>`, allowing `trouble` to type check. But during *monomorphization*,
`weaponize` will actually produce a version of the code that returns a boolean
instead, and then attempt to dereference that boolean. In other words, things
look different to the typechecker than they do to codegen. Oops.

So what went wrong? It should be fine for the compiler to assume that `T:
Example` for all `T`, given the blanket impl. But it's clearly problematic to
*also* assume that the associated types will be the ones given by that blanket
impl. Thus, the "obvious" solution is just to generate a type error in `trouble`
by preventing it from assuming `<T as Example>::Output` is `Box<T>`.

Unfortunately, this solution doesn't work. For one thing, it would be a breaking
change, since the following code *does* compile today:

```rust
trait Example {
    type Output;
    fn generate(self) -> Self::Output;
}

impl<T> Example for T {
    type Output = Box<T>;
    fn generate(self) -> Box<T> { Box::new(self) }
}

fn trouble<T>(t: T) -> Box<T> {
    Example::generate(t)
}
```

And there are definitely cases where this pattern is important. To pick just one
example, consider the following impl for the slice iterator:

```rust
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    // ...
}
```

It's essential that downstream code be able to assume that `<Iter<'a, T> as
Iterator>::Item` is just `&'a T`, no matter what `'a` and `T` happen to be.

Furthermore, it doesn't work to say that the compiler can make this kind of
assumption *unless* specialization is being used, since we want to allow
downstream crates to add specialized impls. We need to know up front.

Another possibility would be to simply disallow specialization of associated
types. But the trouble described above isn't limited to associated types. Every
function/method in a trait has an implicit associated type that implements the
closure types, and similar bad assumptions about blanket impls can crop up
there. It's not entirely clear whether they can be weaponized, however. (That
said, it may be reasonable to stabilize only specialization of functions/methods
to begin with, and wait for strong use cases of associated type specialization
to emerge before stabilizing that.)

The solution proposed in this RFC is instead to treat specialization of items in
a trait as a per-item *opt in*, described in the next section. (This opt in, it
should be noted, is desirable for other reasons as well.)

## The `default` keyword

Many statically-typed languages that allow refinement of behavior in some
hierarchy also come with ways to signal whether or not this is allowed:

- C++ requires the `virtual` keyword to permit a method to be overridden in
  subclasses. Modern C++ also supports `final` and `override` qualifiers.

- C# requires the `virtual` keyword at definition and `override` at point of
  overriding an existing method.

- Java makes things silently virtual, but supports `final` as an opt out.

Why have these qualifiers? Overriding implementations is, in a way, "action at a
distance". It means that the code that's actually being run isn't obvious when
e.g. a class is defined; it can change in subclasses defined
elsewhere. Requiring qualifiers is a way of signaling that this non-local change
is happening, so that you know you need to look more globally to understand the
actual behavior of the class.

While impl specialization does not directly involve virtual dispatch, it's
closely-related to inheritance, and it allows some amount of "action at a
distance" (modulo, as we'll see, coherence rules). We can thus borrow directly
from these previous designs.

This RFC proposes a "final-by-default" semantics akin to C++ that is
backwards-compatible with today's Rust, which means that the following
overlapping impls are prohibited:

```rust
impl<T> Example for T {
    type Output = Box<T>;
    fn generate(self) -> Box<T> { Box::new(self) }
}

impl Example for bool {
    type Output = bool;
    fn generate(self) -> bool { self }
}
```

The error in these impls is that the first impl is implicitly defining "final"
versions of its items, which are thus not allowed to be refined in further
specializations.

If you want to allow specialization of an item, you do so via the `default`
qualifier *within the impl block*:

```rust
impl<T> Example for T {
    default type Output = Box<T>;
    default fn generate(self) -> Box<T> { Box::new(self) }
}

impl Example for bool {
    type Output = bool;
    fn generate(self) -> bool { self }
}
```

Thus, when you're trying to understand what code is going to be executed, if you
see an impl that applies to a type and the relevant item is *not* marked
`default`, you know that the definition you're looking at is the one that will
apply. If, on the other hand, the item is marked `default`, you need to scan for
other impls that could apply to your type. The coherence rules, described below,
help limit the scope of this search in practice.

This design optimizes for fine-grained control over when specialization is
permitted. It's worth pausing for a moment and considering some alternatives and
questions about the design:

- **Why mark `default` on impls rather than the trait?** There are a few reasons
  to have `default` apply at the impl level. First of all, traits are
  fundamentally *interfaces*, while `default` is really about
  *implementations*. Second, as we'll see, it's useful to be able to "seal off"
  a certain avenue of specialization while leaving others open; doing it at the
  trait level is an all-or-nothing choice.

- **Why mark `default` on items rather than the entire impl?** Again, this is
  largely about granularity; it's useful to be able to pin down part of an impl
  while leaving others open for specialization. Furthermore, while this RFC
  doesn't propose to do it, we could easily add a shorthand later on in which
  `default impl Trait for Type` is sugar for adding `default` to all items in
  the impl.

- **Won't `default` be confused with default methods?** Yes! But usefully so: as
  we'll see, in this RFC's design today's default methods become sugar for
  tomorrow's specialization.

Finally, how does `default` help with the hazards described above? Easy: an
associated type from a blanket impl must be treated "opaquely" if it's marked
`default`. That is, if you write these impls:

```rust
impl<T> Example for T {
    default type Output = Box<T>;
    default fn generate(self) -> Box<T> { Box::new(self) }
}

impl Example for bool {
    type Output = bool;
    fn generate(self) -> bool { self }
}
```

then the function `trouble` will fail to typecheck:

```rust
fn trouble<T>(t: T) -> Box<T> {
    Example::generate(t)
}
```

The error is that `<T as Example>::Output` no longer normalizes to `Box<T>`,
because the applicable blanket impl marks the type as `default`. The fact that
`default` is an opt in makes this behavior backwards-compatible.

The main drawbacks of this solution are:

- **API evolution**. Adding `default` to an associated type *takes away* some
  abilities, which makes it a breaking change to a public API. (In principle,
  this is probably true for functions/methods as well, but the breakage there is
  theoretical at most.) However, given the design constraints discussed so far,
  this seems like an inevitable aspect of any simple, backwards-compatible
  design.

- **Verbosity**. It's possible that certain uses of the trait system will result
  in typing `default` quite a bit. This RFC takes a conservative approach of
  introducing the keyword at a fine-grained level, but leaving the door open to
  adding shorthands (like writing `default impl ...`) in the future, if need be.

## Overlapping impls and specialization

### What is overlap?

Rust today does not allow any "overlap" between impls. Intuitively, this means
that you cannot write two trait impls that could apply to the same "input"
types. (An input type is either `Self` or a type parameter of the trait). For
overlap to occur, the input types must be able to "unify", which means that
there's some way of instantiating any type parameters involved so that the input
types are the same. Here are some examples:

```rust
trait Foo {}

// No overlap: String and Vec<u8> cannot unify.
impl Foo for String {}
impl Foo for Vec<u8> {}

// No overlap: Vec<u16> and Vec<u8> cannot unify because u16 and u8 cannot unify.
impl Foo for Vec<u16> {}
impl Foo for Vec<u8> {}

// Overlap: T can be instantiated to String.
impl<T> Foo for T {}
impl Foo for String {}

// Overlap: Vec<T> and Vec<u8> can unify because T can be instantiated to u8.
impl<T> Foo for Vec<T> {}
impl Foo for Vec<u8>

// No overlap: String and Vec<T> cannot unify, no matter what T is.
impl Foo for String {}
impl<T> Foo for Vec<T> {}

// Overlap: for any T that is Clone, both impls apply.
impl<T> Foo for Vec<T> where T: Clone {}
impl<T> Foo for Vec<T> {}

// No overlap: implicitly, T: Sized, and since !Foo: Sized, you cannot instantiate T with it.
impl<T> Foo for Box<T> {}
impl Foo for Box<Foo> {}

trait Trait1 {}
trait Trait2 {}

// Overlap: nothing prevents a T such that T: Trait1 + Trait2.
impl<T: Trait1> Foo for T {}
impl<T: Trait2> Foo for T {}

trait Trait3 {}
trait Trait4: Trait3 {}

// Overlap: any T: Trait4 is covered by both impls.
impl<T: Trait3> Foo for T {}
impl<T: Trait4> Foo for T {}

trait Bar<T> {}

// No overlap: *all* input types must unify for overlap to happen.
impl Bar<u8> for u8 {}
impl Bar<u16> for u8 {}

// No overlap: *all* input types must unify for overlap to happen.
impl<T> Bar<u8> for T {}
impl<T> Bar<u16> for T {}

// No overlap: no way to instantiate T such that T == u8 and T == u16.
impl<T> Bar<T> for T {}
impl Bar<u16> for u8 {}

// Overlap: instantiate U as T.
impl<T> Bar<T> for T {}
impl<T, U> Bar<T> for U {}

// No overlap: no way to instantiate T such that T == &'a T.
impl<T> Bar<T> for T {}
impl<'a, T> Bar<&'a T> for T {}

// Overlap: instantiate T = &'a U.
impl<T> Bar<T> for T {}
impl<'a, T, U> Bar<T> for &'a U where U: Bar<T> {}
```

### Permitting overlap

The goal of specialization is to allow overlapping impls, but it's not as simple
as permitting *all* overlap. There has to be a way to decide which of several
overlapping impls to actually use for a given set of input types. The simpler
and more intuitive the rule for deciding, the easier it is to write and reason
about code -- and since dispatch is already quite complicated, simplicity here
is a high priority. On the other hand, the design should support as many of the
motivating use cases as possible.

The basic intuition we've been using for specialization is the idea that one
impl can be "more specific" than another, and that for any given type there
should be at most one "most specific" impl that applies.

Before turning this intuition into a rule, let's go through the previous
examples of overlap and decide which, if any, of the impls is intuitively more
specific. **Note that since we're leaving out the body of the impls, you won't
see the `default` keyword that would be required in practice for the less
specialized impls.**

```rust
trait Foo {}

// Overlap: T can be instantiated to String.
impl<T> Foo for T {}
impl Foo for String {}          // String is more specific than T

// Overlap: Vec<T> and Vec<u8> can unify because T can be instantiated to u8.
impl<T> Foo for Vec<T> {}
impl Foo for Vec<u8>            // Vec<u8> is more specific than Vec<T>

// Overlap: for any T that is Clone, both impls apply.
impl<T> Foo for Vec<T>          // "Vec<T> where T: Clone" is more specific than "Vec<T> for any T"
    where T: Clone {}
impl<T> Foo for Vec<T> {}

trait Trait1 {}
trait Trait2 {}

// Overlap: nothing prevents a T such that T: Trait1 + Trait2
impl<T: Trait1> Foo for T {}    // Neither is more specific;
impl<T: Trait2> Foo for T {}    // there's no relationship between the traits here

trait Trait3 {}
trait Trait4: Trait3 {}

// Overlap: any T: Trait4 is covered by both impls.
impl<T: Trait3> Foo for T {}
impl<T: Trait4> Foo for T {}    // T: Trait4 is more specific than T: Trait3

trait Bar<T> {}

// Overlap: instantiate U as T.
impl<T> Bar<T> for T {}         // More specific since both input types are identical
impl<T, U> Bar<T> for U {}

// Overlap: instantiate T = &'a U.
impl<T> Bar<T> for T {}         // Neither is more specific
impl<'a, T, U> Bar<T> for &'a U
    where U: Bar<T> {}
```

What are the patterns here?

- Concrete types are more specific than type variables, e.g.:
  - `String` is more specific than `T`
  - `Vec<u8>` is more specific than `Vec<T>`
- More constraints lead to more specific impls, e.g.:
  - `T: Clone` is more specific than `T`
  - `Bar<T> for T` is more specific than `Bar<T> for U`
- Unrelated constraints don't contribute, e.g.:
  - Neither `T: Trait1` nor `T: Trait2` is more specific than the other.

For many purposes, the above simple patterns are sufficient for working with
specialization. But to provide a spec, we need a more general, formal way of
deciding precedence; we'll give one next.

### Defining the precedence rules

An impl block `I` contains basically two pieces of information relevant to
specialization:

- A set of type variables, like `T, U` in `impl<T, U> Bar<T> for U`.
  - We'll call this `I.vars`.
- A set of where clauses, like `T: Clone` in `impl<T: Clone> Foo for Vec<T>`.
  - We'll call this `I.wc`.

We're going to define a *specialization relation* `<=` between impl blocks, so
that `I <= J` means that impl block `I` is "at least as specific as" impl block
`J`. (If you want to think of this in terms of "size", you can imagine that the
set of types `I` applies to is no bigger than those `J` applies to.)

We'll say that `I < J` if `I <= J` and `!(J <= I)`. In this case, `I` is *more
specialized* than `J`.

To ensure specialization is coherent, we will ensure that for any two impls `I`
and `J` that overlap, there must be an impl that is *precisely their
intersection*. That intersecting impl might just *be* one of `I` or `J` -- in
other words, the rule is automatically satisfied if `I < J` or `J < I`.

For example:

```rust
trait Foo {}
trait Trait1 {}
trait Trait2 {}
trait Trait3 {}

// these two impls overlap without one being more specific than the other:
impl<T> Foo for T where T: Trait1 {}
impl<T> Foo for T where T: Trait2 {}

// ... but this one gives their intersection:
impl<T> Foo for T where T: Trait1 + Trait2 {}

// ... and this one is just more specific than all the others
impl<T> Foo for T where T: Trait1 + Trait2 + Trait3 {}
```

Note in particular that the intersection of the last two impls *is* the last
impl.

This rule guarantees that, given any concrete type that has at least one
applicable impl, we'll be able to find a *single most-specific impl* that
applies. In other words if you take the set of all applicable impls
`ALL_APPLICABLE_IMPLS`:

- There will be some `I` in `ALL_APPLICABLE_IMPLS` such that:
  - For all `J` in `ALL_APPLICABLE_IMPLS` such that:
    - `I <= J`

And this most-specific impl is what we'll dispatch to.

(It's worth pausing to think for a moment how this rule can lead to dispatching
to each of the impls in the example above.)

One nice thing about this approach is that, if there is an overlap without there
being an intersecting impl, the compiler can tell the programmer *precisely
which impl needs to be written* to disambiguate the overlapping portion.

We'll start with an abstract/high-level formulation, and then build up toward an
algorithm for deciding specialization by introducing a number of building
blocks.

#### Abstract formulation

Recall that the
[input types](https://github.com/aturon/rfcs/blob/associated-items/active/0000-associated-items.md)
of a trait are the `Self` type and all trait type parameters. So the following
impl has input types `bool`, `u8` and `String`:

```rust
trait Baz<X, Y> { .. }
// impl I
impl Baz<bool, u8> for String { .. }
```

If you think of these input types as a tuple, `(bool, u8, String`) you can think
of each trait impl `I` as determining a set `apply(I)` of input type tuples that
obeys `I`'s where clauses. The impl above is just the singleton set `apply(I) = { (bool,
u8, String) }`.  Here's a more interesting case:

```rust
// impl J
impl<T, U> Baz<T, u8> for U where T: Clone { .. }
```

which gives the set `apply(J) = { (T, u8, U) | T: Clone }`.

Two impls `I` and `J` overlap if `apply(I)` and `apply(J)` intersect.

**We can now define the specialization order abstractly**: `I <= J` if
`apply(I)` is a subset of `apply(J)`.

This is true of the two sets above:

```
apply(I) = { (bool, u8, String) }
  is a strict subset of
apply(J) = { (T, u8, U) | T: Clone }
```

Here are a few more examples.

**Via where clauses**:

```rust
// impl I
// apply(I) = { T | T a type }
impl<T> Foo for T {}

// impl J
// apply(J) = { T | T: Clone }
impl<T> Foo for T where T: Clone {}

// J < I
```

**Via type structure**:

```rust
// impl I
// apply(I) = { (T, U) | T, U types }
impl<T, U> Bar<T> for U {}

// impl J
// apply(J) = { (T, T) | T a type }
impl<T> Bar<T> for T {}

// J < I
```

The same reasoning can be applied to all of the examples we saw earlier, and the
reader is encouraged to do so. We'll look at some of the more subtle cases here:

```rust
trait Foo {}
trait Trait1 {}
trait Trait2 {}
trait Trait3 {}

// impl I
// apply(I) = { T | T: Trait1 }
impl<T> Foo for T where T: Trait1 {}

// impl J
// apply(J) = { T | T: Trait2 }
impl<T> Foo for T where T: Trait2 {}

// Neither I < J or J < I, but:

// impl K
// apply(K) = { T | T: Trait1, T: Trait2 }
impl<T> Foo for T where T: Trait1 + Trait2 {}

// apply(I) intersect apply(J) = apply(K)
// apply(I) intersect apply(K) = apply(K)
// apply(J) intersect apply(K) = apply(K)
```

```rust
// impl I
// apply(I) = { (T, T) | T any type }
impl<T> Bar<T> for T {}

// impl J
// apply(J) = { (T, &'a U) | U: Bar<T>, 'a any lifetime }
impl<'a, T, U> Bar<T> for &'a U where U: Bar<T> {}
```

The claim is that `apply(I)` and `apply(J)` intersect, but neither contains the
other. Thus, for these impls to coexist, there must be a *third* impl for which
`apply` is precisely the intersection:

```rust
// impl I
// apply(I) = { (T, T) | T any type }
impl<T> Bar<T> for T {}

// impl J
// apply(J) = { (T, &'a U) | U: Bar<T>, 'a any lifetime }
impl<'a, T, U> Bar<T> for &'a U where U: Bar<T> {}

// impl K
// apply(K) = { (&'a T, &'a T) | T: Bar<&'a T>, 'a any lifetime }
impl<'a, T> Bar<&'a T> for &'a T where T: Bar<&'a T> {}

// apply(I) intersect apply(J) = apply(K)
// apply(I) intersect apply(K) = apply(K)
// apply(J) intersect apply(K) = apply(K)
```

#### Algorithmic formulation

The goal in the remainder of this section is to turn the above abstract
definition of `<=` into something closer to an algorithm, connected to existing
mechanisms in the Rust compiler. We'll start by reformulating `<=` in a way that
effectively "inlines" `apply`:

`I <= J` if:

- For any way of instantiating `I.vars`, there is some way of instantiating
  `J.vars` such that the `Self` type and trait type parameters match up.

- For this instantiation of `I.vars`, if you assume `I.wc` holds, you can prove
  `J.wc`.

It turns out that the compiler is already quite capable of answering these
questions, via "unification" and "skolemization", which we'll see next.

##### Unification: solving equations on types

Unification is the workhorse of type inference and many other mechanisms in the
Rust compiler. You can think of it as a way of solving equations on types that
contain variables. For example, consider the following situation:

```rust
fn use_vec<T>(v: Vec<T>) { .. }

fn caller() {
    let v = vec![0u8, 1u8];
    use_vec(v);
}
```

The compiler ultimately needs to infer what type to use for the `T` in `use_vec`
within the call in `caller`, given that the actual argument has type
`Vec<u8>`. You can frame this as a unification problem: solve the equation
`Vec<T> = Vec<u8>`. Easy enough: `T = u8`!

Some equations can't be solved. For example, if we wrote instead:

```rust
fn caller() {
    let s = "hello";
    use_vec(s);
}
```

we would end up equating `Vec<T> = &str`. There's no choice of `T` that makes
that equation work out. Type error!

Unification often involves solving a series of equations between types
simultaneously, but it's not like high school algebra; the equations involved
all have the limited form of `type1 = type2`.

One immediate way in which unification is relevant to this RFC is in determining
when two impls "overlap": roughly speaking, they overlap if you can each pair of
input types can be unified simultaneously. For example:

```rust
// No overlap: String and bool do not unify
impl Foo for String { .. }
impl Foo for bool { .. }

// Overlap: String and T unify
impl Foo for String { .. }
impl<T> Foo for T { .. }

// Overlap: T = U, T = V is trivially solvable
impl<T> Bar<T> for T { .. }
impl<U, V> Bar<U> for V { .. }

// No overlap: T = u8, T = bool not solvable
impl<T> Bar<T> for T { .. }
impl Bar<u8> for bool { .. }
```

Note the difference in how *concrete types* and *type variables* work for
unification. When `T`, `U` and `V` are variables, it's fine to say that `T = U`,
`T = V` is solvable: we can make the impls overlap by instantiating all three
variables with the same type. But asking for e.g. `String = bool` fails, because
these are concrete types, not variables. (The same happens in algebra; consider
that `2 = 3` cannot be solved, but `x = y` and `y = z` can be.)  This
distinction may seem obvious, but we'll next see how to leverage it in a
somewhat subtle way.

##### Skolemization: asking forall/there exists questions

We've already rephrased `<=` to start with a "for all, there exists" problem:

- For any way of instantiating `I.vars`, there is some way of instantiating
  `J.vars` such that the `Self` type and trait type parameters match up.

For example:

```rust
// impl I
impl<T> Bar<T> for T {}

// impl J
impl<U,V> Bar<U> for V {}
```

For any choice of `T`, it's possible to choose a `U` and `V` such that the two
impls match -- just choose `U = T` and `V = T`. But the opposite isn't possible:
if `U` and `V` are different (say, `String` and `bool`), then no choice of `T`
will make the two impls match up.

This feels similar to a unification problem, and it turns out we can solve it
with unification using a scary-sounding trick known as "skolemization".

Basically, to "skolemize" a type variable is to treat it *as if it were a
concrete type*. So if `U` and `V` are skolemized, then `U = V` is unsolvable, in
the same way that `String = bool` is unsolvable. That's perfect for capturing
the "for any instantiation of I.vars" part of what we want to formalize.

With this tool in hand, we can further rephrase the "for all, there exists" part
of `<=` in the following way:

- After skolemizing `I.vars`, it's possible to unify `I` and `J`.

Note that a successful unification through skolemization gives you the same
answer as you'd get if you unified without skolemizing.

##### The algorithmic version

One outcome of running unification on two impls as above is that we can
understand both impl headers in terms of a single set of type variables. For
example:

```rust
// Before unification:
impl<T> Bar<T> for T where T: Clone { .. }
impl<U, V> Bar<U> for Vec<V> where V: Debug { .. }

// After unification:
// T = Vec<W>
// U = Vec<W>
// V = W
impl<W> Bar<Vec<W>> for Vec<W> where Vec<W>: Clone { .. }
impl<W> Bar<Vec<W>> for Vec<W> where W: Debug { .. }
```

By putting everything in terms of a single set of type params, it becomes
possible to do things like compare the `where` clauses, which is the last piece
we need for a final rephrasing of `<=` that we can implement directly.

Putting it all together, we'll say `I <= J` if:

- After skolemizing `I.vars`, it's possible to unify `I` and `J`.
- Under the resulting unification, `I.wc` implies `J.wc`

Let's look at a couple more examples to see how this works:

```rust
trait Trait1 {}
trait Trait2 {}

// Overlap: nothing prevents a T such that T: Trait1 + Trait2
impl<T: Trait1> Foo for T {}    // Neither is more specific;
impl<T: Trait2> Foo for T {}    // there's no relationship between the traits here
```

In comparing these two impls in either direction, we make it past unification
and must try to prove that one where clause implies another. But `T: Trait1`
does not imply `T: Trait2`, nor vice versa, so neither impl is more specific
than the other. Since the impls do overlap, there must be a third impl for their
intersection (`T: Trait1 + Trait2`).

On the other hand:

```rust
trait Trait3 {}
trait Trait4: Trait3 {}

// Overlap: any T: Trait4 is covered by both impls.
impl<T: Trait3> Foo for T {}
impl<T: Trait4> Foo for T {}    // T: Trait4 is more specific than T: Trait3
```

Here, since `T: Trait4` implies `T: Trait3` but not vice versa, we get

```rust
impl<T: Trait4> Foo for T    <    impl<T: Trait3> Foo for T
```

##### Key properties

For each pair of impls `I`, `J`, the compiler will check that exactly one of the
following holds:

- `I` and `J` do not overlap (a unification check), or else
- There is some impl `K` such that `apply(K) = apply(I) intersect apply(J)`

Recall also that if there is an overlap without there being an intersecting
impl, the compiler can tell the programmer *precisely which impl needs to be
written* to disambiguate the overlapping portion.

Since `I <= J` ultimately boils down to a subset relationship, we get a lot of
nice properties for free (e.g., transitivity: if `I <= J <= K` then `I <= K`).
Together with the compiler check above, we know that at monomorphization time,
after filtering to the impls that apply to some concrete input types, there will
always be a unique, smallest impl in specialization order. (In particular, if
multiple impls apply to concrete input types, those impls must overlap.)

There are various implementation strategies that avoid having to recalculate the
ordering during monomorphization, but we won't delve into those details in this
RFC.

### Implications for coherence

The coherence rules ensure that there is never an ambiguity about which impl to
use when monomorphizing code. Today, the rules consist of the simple overlap
check described earlier, and the "orphan" check which limits the crates in which
impls are allowed to appear ("orphan" refers to an impl in a crate that defines
neither the trait nor the types it applies to). The orphan check is needed, in
particular, so that overlap cannot be created accidentally when linking crates
together.

The design in this RFC heavily revises the overlap check, as described above,
but does not propose any changes to the orphan check (which is described in
[a blog post](http://smallcultfollowing.com/babysteps/blog/2015/01/14/little-orphan-impls/)). Basically,
the change to the overlap check does not appear to change the cases in which
orphan impls can cause trouble. And a moment's thought reveals why: if two
sibling crates are unaware of each other, there's no way that they could each
provide an impl overlapping with the other, yet be sure that one of those impls
is more specific than the other in the overlapping region.

## Partial impls

An interesting consequence of specialization is that impls need not (and in fact
sometimes *cannot*) provide all of the items that a trait specifies. Of course,
this is already the case with defaulted items in a trait -- but as we'll see,
that mechanism can be seen as just a way of using specialization.

Let's start with a simple example:

```rust
trait MyTrait {
    fn foo(&self);
    fn bar(&self);
}

impl<T: Clone> MyTrait for T {
    default fn foo(&self) { ... }
    default fn bar(&self) { ... }
}

impl MyTrait for String {
    fn bar(&self) { ... }
}
```

Here, we're acknowledging that the blanket impl has already provided definitions
for both methods, so the impl for `String` can opt to just re-use the earlier
definition of `foo`. This is one reason for the choice of the keyword `default`.
Viewed this way, items defined in a specialized impl are optional overrides of
those in overlapping blanket impls.

And, in fact, if we'd written the blanket impl differently, we could *force* the
`String` impl to leave off `foo`:

```rust
impl<T: Clone> MyTrait for T {
    // now `foo` is "final"
    fn foo(&self) { ... }

    default fn bar(&self) { ... }
}
```

Being able to leave off items that are covered by blanket impls means that
specialization is close to providing a finer-grained version of defaulted items
in traits -- one in which the defaults can become ever more refined as more is
known about the input types to the traits (as described in the Motivation
section). But to fully realize this goal, we need one other ingredient: the
ability for the *blanket* impl itself to leave off some items. We do this by
using the `partial` keyword:

```rust
trait Add<Rhs=Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
    fn add_assign(&mut self, Rhs);
}

partial impl<T: Clone, Rhs> Add<Rhs> for T {
    fn add_assign(&mut self, rhs: R) {
        let tmp = self.clone() + rhs;
        *self = tmp;
    }
}
```

A subsequent overlapping impl of `Add` where `Self: Clone` can choose to leave
off `add_assign`, "inheriting" it from the partial impl above.

A key point here is that, as the keyword suggests, a `partial` impl may be
incomplete: from the above code, you *cannot* assume that `T: Add<T>` for any
`T: Clone`, because no such complete impl has been provided.

With partial impls, defaulted items in traits are just sugar for a partial
blanket impl:

```rust
trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
    // ...
}

// desugars to:

trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
    // ...
}

partial impl<T> Iterator for T {
    default fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
    // ...
}
```

Partial impls are somewhat akin to abstract base classes in object-oriented
languages; they provide some, but not all, of the materials needed for a fully
concrete implementation, and thus enable code reuse but cannot be used concretely.

Note that partial impls still need to use `default` to allow for overriding --
leaving off the qualifier will lock down the implementation in any
more-specialized complete impls, which is actually a useful pattern (as
explained in the Motivation.)

There are a few important details to nail down with the design. This RFC
proposes starting with the conservative approach of applying the general overlap
rule to partial impls, same as with complete ones. That ensures that there is
always a clear definition to use when providing subsequent complete impls.  It
would be possible, though, to relax this constraint and allow *arbitrary*
overlap between partial impls, requiring then whenever a complete impl overlaps
with them, *for each item*, there is either a unique "most specific" partial
impl that applies, or else the complete impl provides its own definition for
that item. Such a relaxed approach is much more flexible, probably easier to
work with, and can enable more code reuse -- but it's also more complicated, and
backwards-compatible to add on top of the proposed conservative approach.

## Possible extensions

It's worth briefly mentioning a couple of mechanisms that one could consider
adding on top of specialization.

### Inherent impls

It has long been folklore that inherent impls can be thought of as special,
anonymous traits that are:

- Automatically in scope;
- Given higher dispatch priority than normal traits.

It is easiest to make this idea work out if you think of each inherent item as
implicitly defining and implementing its own trait, so that you can account for
examples like the following:

```rust
struct Foo<T> { .. }

impl<T> Foo<T> {
    fn foo(&self) { .. }
}

impl<T: Clone> Foo<T> {
    fn bar(&self) { .. }
}
```

In this example, the availability of each inherent item is dependent on a
distinct `where` clause. A reasonable "desugaring" would be:

```rust
#[inherent] // an imaginary attribute turning on the "special" treatment of inherent impls
trait Foo_foo<T> {
    fn foo(&self);
}

#[inherent]
trait Foo_bar<T> {
    fn bar(&self);
}

impl<T> Foo_foo<T> for Foo<T> {
    fn foo(&self) { .. }
}

impl<T: Clone> Foo_bar<T> for Foo<T> {
    fn bar(&self) { .. }
}
```

With this idea in mind, it is natural to expect specialization to work for
inherent impls, e.g.:

```rust
impl<T, I> Vec<T> where I: IntoIterator<Item = T> {
    default fn extend(iter: I) { .. }
}

impl<T> Vec<T> {
    fn extend(slice: &[T]) { .. }
}
```

We could permit such specialization at the inherent impl level. The
semantics would be defined in terms of the folklore desugaring above.

(Note: this example was chosen purposefully: it's possible to use specialization
at the inherent impl level to avoid refactoring the `Extend` trait as described
in the Motivation section.)

There are more details about this idea in the appendix.

### Super

Continuing the analogy between specialization and inheritance, one could imagine
a mechanism like `super` to access and reuse less specialized implementations
when defining more specialized ones. While there's not a strong need for this
mechanism as part of this RFC, it's worth checking that the specialization
approach is at least compatible with `super`.

Fortunately, it is. If we take `super` to mean "the most specific impl
overlapping with this one", there is always a unique answer to that question,
because all overlapping impls are totally ordered with respect to each other via
specialization.

### Extending HRTBs

In the Motivation we mentioned the need to refactor the `Extend` trait to take
advantage of specialization. It's possible to work around that need by using
specialization on inherent impls (and having the trait impl defer to the
inherent one), but of course that's a bit awkward.

For reference, here's the refactoring:

```rust
// Current definition
pub trait Extend<A> {
    fn extend<T>(&mut self, iterable: T) where T: IntoIterator<Item=A>;
}

// Refactored definition
pub trait Extend<A, T: IntoIterator<Item=A>> {
    fn extend(&mut self, iterable: T);
}
```

One problem with this kind of refactoring is that you *lose* the ability to say
that a type `T` is extendable *by an arbitrary iterator*, because every use of
the `Extend` trait has to say precisely what iterator is supported. But the
whole point of this exercise is to have a blanket impl of `Extend` for any
iterator that is then specialized later.

This points to a longstanding limitation: the trait system makes it possible to
ask for any number of specific impls to exist, but not to ask for a blanket impl
to exist -- *except* in the limited case of lifetimes, where higher-ranked trait
bounds allow you to do this:

```rust
trait Trait { .. }
impl<'a> Trait for &'a MyType { .. }

fn use_all<T>(t: T) where for<'a> &'a T: Trait { .. }
```

We could extend this mechanism to cover type parameters as well, so that you could write:

```rust
fn needs_extend_all<T>(t: T) where for<I: IntoIterator<Item=u8>> T: Extend<u8, I> { .. }
```

Such a mechanism is out of scope for this RFC.

# Drawbacks

Many of the more minor tradeoffs have been discussed in detail throughout. We'll
focus here on the big picture.

As with many new language features, the most obvious drawback of this proposal
is the increased complexity of the language -- especially given the existing
complexity of the trait system. Partly for that reason, the RFC errs on the side
of simplicity in the design wherever possible.

One aspect of the design that mitigates its complexity somewhat is the fact that
it is entirely opt in: you have to write `default` in an impl in order for
specialization of that item to be possible. That means that all the ways we have
of reasoning about existing code still hold good. When you do opt in to
specialization, the "obviousness" of the specialization rule should mean that
it's easy to tell at a glance which of two impls will be preferred.

On the other hand, the simplicity of this design has its own drawbacks:

- You have to lift out trait parameters to enable specialization, as in the
  `Extend` example above. The RFC mentions a few ways of dealing with this
  limitation -- either by employing inherent item specialization, or by
  eventually generalizing HRTBs.

- You can't use specialization to handle some of the more "exotic" cases of
  overlap, as described in the Limitations section above. This is a deliberate
  trade, favoring simple rules over maximal expressiveness.

Finally, if we take it as a given that we want to support some form of
"efficient inheritance" as at least a programming pattern in Rust, the ability
to use specialization to do so, while also getting all of its benefits, is a net
simplifier. The full story there, of course, depends on the forthcoming companion RFC.

# Alternatives

## Alternatives to specialization

The main alternative to specialization in general is an approach based on
negative bounds, such as the one outlined in an
[earlier RFC](https://github.com/rust-lang/rfcs/pull/586). Negative bounds make
it possible to handle many of the examples this proposal can't (the ones in the
Limitations section). But negative bounds are also fundamentally *closed*: they
make it possible to perform a certain amount of specialization up front when
defining a trait, but don't easily support downstream crates further
specializing the trait impls.

## Alternative specialization designs

### Explicit ordering

Another, perhaps more palatable alternative would be to take the specialization
rule proposed in this RFC, but have some other way of specifying precedence when
that rule can't resolve it -- perhaps by explicit priority numbering. That kind
of mechanism is usually noncompositional, but due to the orphan rule, it's a
least a crate-local concern. Like the alternative rule above, it could be added
backwards compatibly if needed, since it only enables new cases.

### Singleton non-default wins

@pnkfelix suggested the following rule, which allows overlap so long as there is
a unique non-default item.

> For any given type-based lookup, either:
>
>  0. There are no results (error)
>
>  1. There is only one lookup result, in which case we're done (regardless of
>     whether it is tagged as default or not),
>
>  2. There is a non-empty set of results with defaults, where exactly one
>     result is non-default -- and then that non-default result is the answer,
>     *or*
>
>  3. There is a non-empty set of results with defaults, where 0 or >1 results
>     are non-default (and that is an error).

This rule is arguably simpler than the one proposed in this RFC, and can
accommodate the examples we've presented throughout. It would also support some
of the cases this RFC cannot, because the default/non-default distinction can be
used to specify an ordering between impls when the subset ordering fails to do
so. For that reason, it is not forward-compatible with the main proposal in this
RFC.

The downsides are:

- Because actual dispatch occurs at monomorphization, errors are generated quite
  late, and only at use sites, not impl sites. That moves traits much more in
  the direction of C++ templates.

- It's less scalable/compositional: this alternative design forces the
  "specialization hierarchy" to be flat, in particular ruling out multiple
  levels of increasingly-specialized blanket impls.

# Unresolved questions

Finally, there are a few important questions not yet addressed by this RFC:

- Presumably, we do not want to permit specialization based on *lifetime*
  parameters, but the algorithm as written does not give them any special
  treatment. That needs to be dealt with in the implementation, at least.

- We've said nothing about the interaction with dropck, which relies on a
  parametricity property for generics that don't have bounds. Specialization
  could potentially be used to subvert that property. That needs to be addressed
  before the RFC can be accepted.

- The design with `default` makes specialization of associated types an
  all-or-nothing affair, but it would occasionally be useful to say that all
  further specializations will at least guarantee some additional trait bound on
  the associated type. This is particularly relevant for the "efficient
  inheritance" use case. Such a mechanism can likely be added, if needed, later
  on.

# Appendix

## More details on inherent impls

One tricky aspect for specializing inherent impls is that, since there is no
explicit trait definition, there is no general signature that each definition of
an inherent item must match. Thinking about `Vec` above, for example, notice
that the two signatures for `extend` look superficially different, although it's
clear that the first impl is the more general of the two.

It's workable to use a very simple-minded conceptual desugaring: each item
desugars into a distinct trait, with type parameters for e.g. each argument and
the return type. All concrete type information then emerges from desugaring into
impl blocks. Thus, for example:

```
impl<T, I> Vec<T> where I: IntoIterator<Item = T> {
    default fn extend(iter: I) { .. }
}

impl<T> Vec<T> {
    fn extend(slice: &[T]) { .. }
}

// Desugars to:

trait Vec_extend<Arg, Result> {
    fn extend(Arg) -> Result;
}

impl<T, I> Vec_extend<I, ()> for Vec<T> where I: IntoIterator<Item = T> {
    default fn extend(iter: I) { .. }
}

impl<T> Vec_extend<&[T], ()> for Vec<T> {
    fn extend(slice: &[T]) { .. }
}
```

All items of a given name must desugar to the same trait, which means that the
number of arguments must be consistent across all impl blocks for a given `Self`
type. In addition, we'd require that *all of the impl blocks overlap* (meaning
that there is a single, most general impl). Without these constraints, we would
implicitly be permitting full-blown overloading on both arity and type
signatures. For the time being at least, we want to restrict overloading to
explicit uses of the trait system, as it is today.

This "desugaring" semantics has the benefits of allowing inherent item
specialization, and also making it *actually* be the case that inherent impls
are really just implicit traits -- unifying the two forms of dispatch. Note that
this is a breaking change, since examples like the following are (surprisingly!)
allowed today:

```rust
struct Foo<A, B>(A, B);

impl<A> Foo<A,A> {
    fn foo(&self, _: u32) {}
}

impl<A,B> Foo<A,B> {
    fn foo(&self, _: bool) {}
}

fn use_foo<A, B>(f: Foo<A,B>) {
    f.foo(true)
}
```

As has been proposed
[elsewhere](https://internals.rust-lang.org/t/pre-rfc-adjust-default-object-bounds/2199/),
this "breaking change" could be made available through a feature flag that must
be used even after stabilization (to opt in to specialization of inherent
impls); the full details will depend on pending revisions to
[RFC 1122](https://github.com/rust-lang/rfcs/pull/1122).
