- Start Date: 2014-11-21
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

This RFC proposes several new *generic conversion* traits. The
motivation is to remove the need for ad hoc conversion traits (like
`FromStr`, `AsSlice`, `ToSocketAddr`, `FromError`) whose *sole role*
is for generics bounds. Aside from cutting down on trait
proliferation, centralizing these traits also helps the ecosystem
avoid incompatible ad hoc conversion traits defined downstream from
the types they convert to or from. It also future-proofs against
eventual language features for ergonomic conversion-based overloading.

# Motivation

The idea of generic conversion traits has come up from
[time](https://github.com/rust-lang/rust/issues/7080)
[to](http://discuss.rust-lang.org/t/pre-rfc-add-a-coerce-trait-to-get-rid-of-the-as-slice-calls/415)
[time](http://discuss.rust-lang.org/t/pre-rfc-remove-fromerror-trait-add-from-trait/783/3),
and now that multidispatch is available they can be made to work
reasonably well. They are worth considering due to the problems they
solve (given below), and considering *now* because they would obsolete
several ad hoc conversion traits (and several more that are in the
pipeline) for `std`.

## Problem 1: overloading over conversions

Rust does not currently support arbitrary, implicit conversions -- and
for some good reasons. However, it is sometimes important
ergonomically to allow a single function to be *explicitly* overloaded
based on conversions.

For example, the
[recently proposed path APIs](https://github.com/rust-lang/rfcs/pull/474)
introduce an `AsPath` trait to make various path operations ergonomic:

```rust
pub trait AsPath for Sized? {
    fn as_path(&self) -> &Path;
}

impl Path {
    ...

    pub fn join<P: AsPath>(&self, path: &P) -> PathBuf { ... }
}
```

The idea in particular is that, given a path, you can join using a
string literal directly. That is:

```rust
// write this:
let new_path = my_path.join("fixed_subdir_name");

// not this:
let new_path = my_path.join(Path::new("fixed_subdir_name"));
```

It's a shame to have to introduce new ad hoc traits every time such an
overloading is desired. And because the traits are ad hoc, it's also
not possible to program generically over conversions themselves.

## Problem 2: duplicate, incompatible conversion traits

There's a somewhat more subtle problem compounding the above: if the
author of the path API neglects to include traits like `AsPath` for
its core types, but downstream crates want to overload on those
conversions, those downstream crates may each introduce their own
conversion traits, which will not be compatible with one another.

Having standard, generic conversion traits cuts down on the total
number of traits, and also ensures that all Rust libraries have an
agreed-upon way to talk about conversions.

## Non-goals

When considering the design of generic conversion traits, it's
tempting to try to do away will *all* ad hoc conversion methods.  That
is, to replace methods like `to_string` and `to_vec` with a single
method `to::<String>` and `to::<Vec<u8>>`.

Unfortunately, this approach carries several ergonomic downsides:

* The required `::< _ >` syntax is pretty unfriendly. Something like
  `to<String>` would be much better, but is unlikely to happen given
  the current grammar.

* Designing the traits to allow this usage is surprisingly subtle --
  it effectively requires *two traits* per type of generic conversion,
  with blanket `impl`s mapping one to the other. Having such
  complexity for *all conversions* in Rust seems like a non-starter.

* Discoverability suffers somewhat. Looking through a method list and
  seeing `to_string` is easier to comprehend (for newcomers
  especially) than having to crawl through the `impl`s for a trait on
  the side -- especially given the trait complexity mentioned above.

Nevertheless, this is a serious alternative that will be laid out in
more detail below, and merits community discussion.

# Detailed design

## Basic design

The design is fairly simple, although perhaps not as simple as one
might expect: we introduce a total of *four* traits:

```rust
trait As<Sized? T> for Sized? {
    fn convert_as(&self) -> &T;
}

trait AsMut<Sized? T> for Sized? {
    fn convert_as_mut(&mut self) -> &mut T;
}

trait To<T> for Sized? {
    fn convert_to(&self) -> T;
}

trait Into<T> {
    fn convert_into(self) -> T;
}

trait From<T> {
    fn from(T) -> Self;
}
```

The first three traits mirror our `as`/`to`/`into` conventions, but
add a bit more structure to them: `as`-style conversions are from
references to references, `to`-style conversions are from references
to arbitrary types, and `into`-style conversions are between arbitrary
types (consuming their argument).

The final trait, `From`, mimics the `from` constructors. Unlike the
other traits, its method is not prefixed with `convert`. This is
because, again unlike the other traits, this trait is expected to
outright replace most custom `from` constructors. See below.

**Why the reference restrictions?**

If all of the conversion traits were between arbitrary types, you
would have to use generalized where clauses and explicit lifetimes even for simple cases:

```rust
// Possible alternative:
trait As<T> {
    fn convert_as(self) -> T;
}

// But then you get this:
fn take_as<'a, T>(t: &'a T) where &'a T: As<&'a MyType>;

// Instead of this:
fn take_as<T>(t: &T) where T: As<MyType>;
```

If you need a conversion that works over any lifetime, you need to use
higher-ranked trait bounds:

```rust
... where for<'a> &'a T: As<&'a MyType>
```

This case is particularly important when you cannot name a lifetime in
advance, because it will be created on the stack within the
function. It might be possible to add sugar so that `where &T:
As<&MyType>` expands to the above automatically, but such an elision
might have other problems, and in any case it would preclude writing
direct bounds like `fn foo<P: AsPath>`.

The proposed trait definition essentially *bakes in* the needed
lifetime connection, capturing the most common mode of use for
`as`/`to`/`into` conversions. In the future, an HKT-based version of
these traits could likely generalize further.

**Why have multiple traits at all**?

The biggest reason to have multiple traits is to take advantage of the
lifetime linking explained above. In addition, however, it is a basic
principle of Rust's libraries that conversions are distinguished by
cost and consumption, and having multiple traits makes it possible to
(by convention) restrict attention to e.g. "free" `as`-style conversions
by bounding only by `As`.

Why have both `Into` and `From`? There are a few reasons:

* Coherence issues: the order of the types is significant, so `From`
  allows extensibility in some cases that `Into` does not.

* To match with existing conventions around conversions and
  constructors (in particular, replacing many `from` constructors).

## Blanket `impl`s

Given the above trait design, there are a few straightforward blanket
`impl`s as one would expect:

```rust
// As implies To
impl<'a, Sized? T, Sized? U> To<&'a U> for &'a T where T: As<U> {
    fn convert_to(&self) -> &'a U {
        self.convert_as()
    }
}

// To implies Into
impl<'a, T, U> Into<U> for &'a T where T: To<U> {
    fn convert_into(self) -> U {
        self.convert_to()
    }
}

// AsMut implies Into
impl<'a, T, U> Into<&'a mut U> for &'a mut T where T: AsMut<U> {
    fn convert_into(self) -> &'a mut U {
        self.convert_as_mut()
    }
}

// Into implies From
impl<T, U> From<T> for U where T: Into<U> {
    fn from(t: T) -> U { t.cvt_into() }
}
```

The interaction between

## An example

Using all of the above, here are some example `impl`s and their use:

```rust
impl As<str> for String {
    fn convert_as(&self) -> &str {
        self.as_slice()
    }
}
impl As<[u8]> for String {
    fn convert_as(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl Into<Vec<u8>> for String {
    fn convert_into(self) -> Vec<u8> {
        self.into_bytes()
    }
}

fn main() {
    let a = format!("hello");
    let b: &[u8] = a.convert_as();
    let c: &str = a.convert_as();
    let d: Vec<u8> = a.convert_into();
}
```

This use of generic conversions within a function body is expected to
be rare, however; usually the traits are used for generic functions:

```
impl Path {
    fn join_path_inner(&self, p: &Path) -> PathBuf { ... }

    pub fn join_path<P: As<Path>>(&self, p: &P) -> PathBuf {
        self.join_path_inner(p.convert_as())
    }
}
```

In this very typical pattern, you introduce an "inner" function that
takes the converted value, and the public API is a thin wrapper around
that. The main reason to do so is to avoid code bloat: given that the
generic bound is used only for a conversion that can be done up front,
there is no reason to monomorphize the entire function body for each
input type.

### An aside: codifying the generics pattern in the language

This pattern is so common that we probably want to consider sugar for
it, e.g. something like:

```rust
impl Path {
    pub fn join_path(&self, p: ~Path) -> PathBuf {
        ...
    }
}
```

that would desugar into exactly the above (assuming that the `~` sigil
was restricted to `As` conversions). Such a feature is out of scope
for this RFC, but it's a natural and highly ergonomic extension of the
traits being proposed here.

## Preliminary conventions

Would *all* conversion traits be replaced by the proposed ones?
Probably not, due to the combination of two factors:

* You still want blanket `impl`s like `ToString` for `Show`, but:
* This RFC proposes that specific conversion *methods* like
  `to_string` stay in common use.

On the other hand, you'd expect a blanket `impl` of `To<String>` for
any `T: ToString`, and one should prefer bounding over `To<String>`
rather than `ToString` for consistency. Basically, the role of
`ToString` is just to provide the ad hoc method name `to_string` in a
blanket fashion.

So a rough, preliminary convention would be the following:

* An *ad hoc conversion method* is one following the normal convention
  of `as_foo`, `to_foo`, `into_foo` or `from_foo`. A "generic"
  conversion method is one going through the generic traits proposed
  in this RFC. An *ad hoc conversion trait* is a trait providing an ad
  hoc conversion method.

* Use ad hoc conversion methods for "natural", *outgoing* conversions
  that should have easy method names and good discoverability. A
  conversion is "natural" if you'd call it directly on the type in
  normal code; "unnatural" conversions usually come from generic
  programming.

  For example, `to_string` is a natural conversion for `str`, while
  `into_string` is not; but the latter is sometimes useful in a
  generic context -- and that's what the generic conversion traits can
  help with.

* On the other hand, favor `From` for all conversion constructors.

* Introduce ad hoc conversion *traits* if you need to provide a
  blanket `impl` of an ad hoc conversion method, or need special
  functionality. For example, `to_string` needs a trait so that every
  `Show` type automatically provides it.

* For any ad hoc conversion method, *also* provide an `impl` of the
  corresponding generic version; for traits, this should be done via a
  blanket `impl`.

* When using generics bounded over a conversion, always prefer to use
  the generic conversion traits. For example, bound `S: To<String>`
  not `S: ToString`. This encourages consistency, and also allows
  clients to take advantage of the various blanket generic conversion
  `impl`s.

* Use the "inner function" pattern mentioned above to avoid code
  bloat.

## Prelude changes

*All* of the conversion traits are added to the prelude. There are two
 reasons for doing so:

* For `As`/`To`/`Into`, the reasoning is similar to the inclusion of
  `PartialEq` and friends: they are expected to appear ubiquitously as
  bounds.

* For `From`, bounds are somewhat less common but the use of the
  `from` constructor is expected to be rather widespread.

# Drawbacks

There are a few drawbacks to the design as proposed:

* Since it does not replace all conversion traits, there's the
  unfortunate case of having both a `ToString` trait and a
  `To<String>` trait bound. The proposed conventions go some distance
  toward at least keeping APIs consistent, but the redundancy is
  unfortunate. See Alternatives for a more radical proposal.

* It may encourage more overloading over coercions, and also more
  generics code bloat (assuming that the "inner function" pattern
  isn't followed). Coercion overloading is not necessarily a bad
  thing, however, since it is still explicit in the signature rather
  than wholly implicit. If we do go in this direction, we can consider
  language extensions that make it ergonomic *and* avoid code bloat.

# Alternatives

The main alternative is one that attempts to provide methods that
*completely replace* ad hoc conversion methods. To make this work, a
form of double dispatch is used, so that the methods are added to
*every type* but bounded by a separate set of conversion traits.

In this strawman proposal, the name "view shift" is used for `as`
conversions, "conversion" for `to` conversions, and "transformation"
for `into` conversions. These names are not too important, but needed
to distinguish the various generic methods.

The punchline is that, in the end, we can write

```rust
let s = format!("hello");
let b = s.shift_view::<[u8]>();
```

or, put differently, replace `as_bytes` with `shift_view::<[u8]>` --
for better or worse.

In addition to the rather large jump in complexity, this alternative
design also suffers from poor error messages. For example, if you
accidentally typed `shift_view::<u8>` instead, you receive:

```
error: the trait `ShiftViewFrom<collections::string::String>` is not implemented for the type `u8`
```

which takes a bit of thought and familiarity with the traits to fully
digest.  Taken together, the complexity, error messages, and poor
ergonomics of things like `convert::<u8>` rather than `as_bytes` led
the author to discard this alternative design.

```rust
// VIEW SHIFTS

// "Views" here are always lightweight, non-lossy, always
// successful view shifts between reference types

// Immutable views

trait ShiftViewFrom<Sized? T> for Sized? {
    fn shift_view_from(&T) -> &Self;
}

trait ShiftView for Sized? {
    fn shift_view<Sized? T>(&self) -> &T where T: ShiftViewFrom<Self>;
}

impl<Sized? T> ShiftView for T {
    fn shift_view<Sized? U: ShiftViewFrom<T>>(&self) -> &U {
        ShiftViewFrom::shift_view_from(self)
    }
}

// Mutable coercions

trait ShiftViewFromMut<Sized? T> for Sized? {
    fn shift_view_from_mut(&mut T) -> &mut Self;
}

trait ShiftViewMut for Sized? {
    fn shift_view_mut<Sized? T>(&mut self) -> &mut T where T: ShiftViewFromMut<Self>;
}

impl<Sized? T> ShiftViewMut for T {
    fn shift_view_mut<Sized? U: ShiftViewFromMut<T>>(&mut self) -> &mut U {
        ShiftViewFromMut::shift_view_from_mut(self)
    }
}

// CONVERSIONS

trait ConvertFrom<Sized? T> for Sized? {
    fn convert_from(&T) -> Self;
}

trait Convert for Sized? {
    fn convert<T>(&self) -> T where T: ConvertFrom<Self>;
}

impl<Sized? T> Convert for T {
    fn convert<U>(&self) -> U where U: ConvertFrom<T> {
        ConvertFrom::convert_from(self)
    }
}

impl ConvertFrom<str> for Vec<u8> {
    fn convert_from(s: &str) -> Vec<u8> {
        s.to_string().into_bytes()
    }
}

// TRANSFORMATION

trait TransformFrom<T> {
    fn transform_from(T) -> Self;
}

trait Transform {
    fn transform<T>(self) -> T where T: TransformFrom<Self>;
}

impl<T> Transform for T {
    fn transform<U>(self) -> U where U: TransformFrom<T> {
        TransformFrom::transform_from(self)
    }
}

impl TransformFrom<String> for Vec<u8> {
    fn transform_from(s: String) -> Vec<u8> {
        s.into_bytes()
    }
}

impl<'a, T, U> TransformFrom<&'a T> for U where U: ConvertFrom<T> {
    fn transform_from(x: &'a T) -> U {
        x.convert()
    }
}

impl<'a, T, U> TransformFrom<&'a mut T> for &'a mut U where U: ShiftViewFromMut<T> {
    fn transform_from(x: &'a mut T) -> &'a mut U {
        ShiftViewFromMut::shift_view_from_mut(x)
    }
}

// Example

impl ShiftViewFrom<String> for str {
    fn shift_view_from(s: &String) -> &str {
        s.as_slice()
    }
}
impl ShiftViewFrom<String> for [u8] {
    fn shift_view_from(s: &String) -> &[u8] {
        s.as_bytes()
    }
}

fn main() {
    let s = format!("hello");
    let b = s.shift_view::<[u8]>();
}
```
