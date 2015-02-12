- Start Date: 2014-10-29
- RFC PR #: [rust-lang/rfcs#235](https://github.com/rust-lang/rfcs/pull/235)
- Rust Issue #: [rust-lang/rust#18424](https://github.com/rust-lang/rust/issues/18424)

# Summary

This is a combined *conventions* and *library stabilization* RFC. The goal is to
establish a set of naming and signature conventions for `std::collections`.

The major components of the RFC include:

* Removing most of the traits in `collections`.

* A general proposal for solving the "equiv" problem, as well as improving
  `MaybeOwned`.

* Patterns for overloading on by-need values and predicates.

* Initial, forwards-compatible steps toward `Iterable`.

* A coherent set of API conventions across the full variety of collections.

*A big thank-you to @Gankro, who helped collect API information and worked
 through an initial pass of some of the proposals here.*

# Motivation

This RFC aims to improve the design of the `std::collections` module in
preparation for API stabilization. There are a number of problems that need to
be addressed, as spelled out in the subsections below.

## Collection traits

The `collections` module defines several traits:

* Collection
* Mutable
* MutableSeq
* Deque
* Map, MutableMap
* Set, MutableSet

There are several problems with the current trait design:

* Most important: the traits do not provide iterator methods like `iter`. It is
  not possible to do so in a clean way without higher-kinded types, as the RFC
  explains in more detail below.

* The split between mutable and immutable traits is not well-motivated by
  any of the existing collections.

* The methods defined in these traits are somewhat anemic compared to the suite
  of methods provided on the concrete collections that implement them.

## Divergent APIs

Despite the current collection traits, the APIs of various concrete collections
has diverged; there is not a globally coherent design, and there are many
inconsistencies.

One problem in particular is the lack of clear guiding principles for the API
design. This RFC proposes a few along the way.

## Providing slice APIs on `Vec` and `String`

The `String` and `Vec` types each provide a limited subset of the methods
provides on string and vector slices, but there is not a clear reason to limit
the API in this way. Today, one has to write things like
`some_str.as_slice().contains(...)`, which is not ergonomic or intuitive.

## The `Equiv` problem

There is a more subtle problem related to slices. It's common to use a `HashMap`
with owned `String` keys, but then the natural API for things like lookup is not
very usable:

```rust
fn find(&self, k: &K) -> Option<&V>
```

The problem is that, since `K` will be `String`, the `find` function requests a
`&String` value -- whereas one typically wants to work with the more flexible
`&str` slices. In particular, using `find` with a literal string requires
something like:

```rust
map.find(&"some literal".to_string())
```

which is unergonomic and requires an extra allocation just to get a borrow that,
in some sense, was already available.

The current `HashMap` API works around this problem by providing an *additional*
set of methods that uses a generic notion of "equivalence" of values that have
different types:

```rust
pub trait Equiv<T> {
    fn equiv(&self, other: &T) -> bool;
}

impl Equiv<str> for String {
    fn equiv(&self, other: &str) -> bool {
        self.as_slice() == other
    }
}

fn find_equiv<Q: Hash<S> + Equiv<K>>(&self, k: &Q) -> Option<&V>
```

There are a few downsides to this approach:

* It requires a duplicated `_equiv` variant of each method taking a reference to
  the key. (This downside could likely be mitigated using
  [multidispatch](https://github.com/rust-lang/rfcs/pull/195).)

* Its correctness depends on equivalent values producing the same hash, which is
  not checked.

* `String`-keyed hash maps are very common, so newcomers are likely to run
  headlong into the problem. First, `find` will fail to work in the expected
  way. But the signature of `find_equiv` is more difficult to understand than
  `find`, and it it's not immediately obvious that it solves the problem.

* It is the right API for `HashMap`, but not helpful for e.g. `TreeMap`, which
  would want an analog for `Ord`.

The `TreeMap` API currently deals with this problem in an entirely different
way:

```rust
/// Returns the value for which f(key) returns Equal.
/// f is invoked with current key and guides tree navigation.
/// That means f should be aware of natural ordering of the tree.
fn find_with(&self, f: |&K| -> Ordering) -> Option<&V>
```

Besides being less convenient -- you cannot write `map.find_with("some literal")` --
this function navigates the tree according to an ordering that may have no
relationship to the actual ordering of the tree.

## `MaybeOwned`

Sometimes a function does not know in advance whether it will need or produce an
owned copy of some data, or whether a borrow suffices. A typical example is the
`from_utf8_lossy` function:

```rust
fn from_utf8_lossy<'a>(v: &'a [u8]) -> MaybeOwned<'a>
```

This function will return a string slice if the input was correctly utf8 encoded
-- without any allocation. But if the input has invalid utf8 characters, the
function allocates a new `String` and inserts utf8 "replacement characters"
instead. Hence, the return type is an `enum`:

```rust
pub enum MaybeOwned<'a> {
    Slice(&'a str),
    Owned(String),
}
```

This interface makes it possible to allocate only when necessary, but the
`MaybeOwned` type (and connected machinery) are somewhat ad hoc -- and
specialized to `String`/`str`. It would be somewhat more palatable if there were
a single "maybe owned" abstraction usable across a wide range of types.

## `Iterable`

A frequently-requested feature for the `collections` module is an `Iterable`
trait for "values that can be iterated over". There are two main motivations:

* *Abstraction*. Today, you can write a function that takes a single `Iterator`,
  but you cannot write a function that takes a container and then iterates over
  it multiple times (perhaps with differing mutability levels). An `Iterable`
  trait could allow that.

* *Ergonomics*. You'd be able to write

  ```rust
  for v in some_vec { ... }
  ```

  rather than

  ```rust
  for v in some_vec.iter() { ... }
  ```

  and `consume_iter(some_vec)` rather than `consume_iter(some_vec.iter())`.

# Detailed design

## The collections today

The concrete collections currently available in `std` fall into roughly three categories:

* Sequences
    * Vec
    * String
    * Slices
    * Bitv
    * DList
    * RingBuf
    * PriorityQueue

* Sets
    * HashSet
    * TreeSet
    * TrieSet
    * EnumSet
    * BitvSet

* Maps
    * HashMap
    * TreeMap
    * TrieMap
    * LruCache
    * SmallIntMap

The primary goal of this RFC is to establish clean and consistent APIs that
apply across each group of collections.

Before diving into the details, there is one high-level changes that should be
made to these collections. The `PriorityQueue` collection should be renamed to
`BinaryHeap`, following the convention that concrete collections are named according
to their implementation strategy, not the abstract semantics they implement. We
may eventually want `PriorityQueue` to be a *trait* that's implemented by
multiple concrete collections.

The `LruCache` could be renamed for a similar reason (it uses a `HashMap` in its
implementation), However, the implementation is actually generic with respect to
this underlying map, and so in the long run (with HKT and other language
changes) `LruCache` should probably add a type parameter for the underlying map,
defaulted to `HashMap`.

## Design principles

* *Centering on `Iterator`s*. The `Iterator` trait is a strength of Rust's
  collections library. Because so many APIs can produce iterators, adding an API
  that consumes one is very powerful -- and conversely as well.  Moreover,
  iterators are highly efficient, since you can chain several layers of
  modification without having to materialize intermediate results.  Thus,
  whenever possible, collection APIs should strive to work with iterators.

  In particular, some existing convenience methods avoid iterators for either
  performance or ergonomic reasons. We should instead improve the ergonomics and
  performance of iterators, so that these extra convenience methods are not
  necessary and so that *all* collections can benefit.

* *Minimizing method variants*. One problem with some of the current collection
  APIs is the proliferation of method variants. For example, `HashMap` include
  *seven* methods that begin with the name `find`! While each method has a
  motivation, the API as a whole can be bewildering, especially to newcomers.

  When possible, we should leverage the trait system, or find other
  abstractions, to reduce the need for method variants while retaining their
  ergonomics and power.

* *Conservatism*. It is easier to add APIs than to take them away.  This RFC
  takes a fairly conservative stance on what should be included in the
  collections APIs. In general, APIs should be very clearly motivated by a wide
  variety of use cases, either for expressiveness, performance, or ergonomics.

## Removing the traits

This RFC proposes a somewhat radical step for the collections traits: rather
than reform them, we should eliminate them altogether -- *for now*.

Unlike inherent methods, which can easily be added and deprecated over time, a
trait is "forever": there are very few backwards-compatible modifications to
traits. Thus, for something as fundamental as collections, it is prudent to take
our time to get the traits right.

### Lack of iterator methods

In particular, there is one way in which the current traits are clearly *wrong*:
they do not provide standard methods like `iter`, despite these being
fundamental to working with collections in Rust. Sadly, this gap is due to
inexpressiveness in the language, which makes directly defining iterator methods
in a trait impossible:

```rust
trait Iter {
    type A;
    type I: Iterator<&'a A>;    // what is the lifetime here?
    fn iter<'a>(&'a self) -> I; // and how to connect it to self?
}
```

The problem is that, when implementing this trait, the return type `I` of `iter`
should depend on the *lifetime* of self. For example, the corresponding
method in `Vec` looks like the following:

```rust
impl<T> Vec<T> {
    fn iter(&'a self) -> Items<'a, T> { ... }
}
```

This means that, given a `Vec<T>`, there isn't a *single* type `Items<T>` for
iteration -- rather, there is a *family* of types, one for each input lifetime.
In other words, the associated type `I` in the `Iter` needs to be
"higher-kinded": not just a single type, but rather a family:

```rust
trait Iter {
    type A;
    type I<'a>: Iterator<&'a A>;
    fn iter<'a>(&self) -> I<'a>;
}
```

In this case, `I` is parameterized by a lifetime, but in other cases (like
`map`) an associated type needs to be parameterized by a type.

In general, such higher-kinded types (HKTs) are a much-requested feature for
Rust. But the design and implementation of higher-kinded types is, by itself, a
significant investment.

HKT would also allow for parameterization over smart pointer types, which has
many potential use cases in the context of collections.

Thus, the goal in this RFC is to do the best we can without HKT *for now*,
while allowing a graceful migration if or when HKT is added.

### Persistent/immutable collections

Another problem with the current collection traits is the split between
immutable and mutable versions. In the long run, we will probably want to
provide *persistent* collections (which allow non-destructive "updates" that
create new collections that share most of their data with the old ones).

However, persistent collection APIs have not been thoroughly explored in Rust;
it would be hasty to standardize on a set of traits until we have more
experience.

### Downsides of removal

There are two main downsides to removing the traits without a replacement:

1. It becomes impossible to write code using generics over a "kind" of
   collection (like `Map`).

2. It becomes more difficult to ensure that the collections share a common API.

For point (1), first, if the APIs are sufficiently consistent it should be
possible to transition code from e.g. a `TreeMap` to a `HashMap` by changing
very few lines of code. Second, generic programming is currently quite limited,
given the inability to iterate. Finally, generic programming over collections is
a large design space (with much precedent in C++, for example), and we should
take our time and gain more experience with a variety of concrete collections
before settling on a design.

For point (2), first, the current traits have failed to keep the APIs in line,
as we will see below. Second, this RFC is the antidote: we establish a clear set
of conventions and APIs for concrete collections up front, and stabilize on
those, which should make it easy to add traits later on.

### Why not leave the traits as "experimental"?

An alternative to removal would be to leave the traits intact, but marked as
experimental, with the intent to radically change them later.

Such a strategy doesn't buy much relative to removal (given the arguments
above), but risks the traits becoming "de facto" stable if people begin using
them en masse.

## Solving the `_equiv` and `MaybeOwned` problems

The basic problem that leads to `_equiv` methods is that:

* `&String` and `&str` are not the same type.
* The `&str` type is more flexible and hence more widely used.
* Code written for a generic type `T` that takes a reference `&T` will therefore
  not be suitable when `T` is instantiated with `String`.

A similar story plays out for `&Vec<T>` and `&[T]`, and with DST and custom
slice types the same problem will arise elsewhere.

### The `Borrow` trait

This RFC proposes to use a *trait*, `Borrow` to connect borrowed and owned data
in a generic fashion:

```rust
/// A trait for borrowing.
trait Borrow<Sized? B> {
    /// Immutably borrow from an owned value.
    fn borrow(&self) -> &B;

    /// Mutably borrow from an owned value.
    fn borrow_mut(&mut self) -> &mut B;
}

// The Sized bound means that this impl does not overlap with the impls below.
impl<T: Sized> Borrow<T> for T {
    fn borrow(a: &T) -> &T {
        a
    }
    fn borrow_mut(a: &mut T) -> &mut T {
        a
    }
}

impl Borrow<str> for String {
    fn borrow(s: &String) -> &str {
        s.as_slice()
    }
    fn borrow_mut(s: &mut String) -> &mut str {
        s.as_mut_slice()
    }
}

impl<T> Borrow<[T]> for Vec<T> {
    fn borrow(s: &Vec<T>) -> &[T] {
        s.as_slice()
    }
    fn borrow_mut(s: &mut Vec<T>) -> &mut [T] {
        s.as_mut_slice()
    }
}
```

*(Note: thanks to @epdtry for [suggesting this variation](https://github.com/rust-lang/rfcs/pull/235#issuecomment-55337168)! The original proposal
 is listed [in the Alternatives](#variants-of-borrow).)*

A primary goal of the design is allowing a *blanket* `impl` for non-sliceable
types (the first `impl` above). This blanket `impl` ensures that all new sized,
cloneable types are automatically borrowable; new `impl`s are required only for
new *unsized* types, which are rare. The `Sized` bound on the initial impl means
that we can freely add impls for unsized types (like `str` and `[T]`) without
running afoul of coherence.

Because of the blanket `impl`, the `Borrow` trait can largely be ignored except
when it is actually used -- which we describe next.

### Using `Borrow` to replace `_equiv` methods

With the `Borrow` trait in place, we can eliminate the `_equiv` method variants
by asking map keys to be `Borrow`:

```rust
impl<K,V> HashMap<K,V> where K: Hash + Eq {
    fn find<Q>(&self, k: &Q) -> &V where K: Borrow<Q>, Q: Hash + Eq { ... }
    fn contains_key<Q>(&self, k: &Q) -> bool where K: Borrow<Q>, Q: Hash + Eq { ... }
    fn insert(&mut self, k: K, v: V) -> Option<V> { ... }

    ...
}
```

The benefits of this approach over `_equiv` are:

* The `Borrow` trait captures the borrowing relationship between an owned data
  structure and both references to it and slices from it -- once and for all.
  This means that it can be used *anywhere* we need to program generically over
  "borrowed" data. In particular, the single trait works for both `HashMap` and
  `TreeMap`, and should work for other kinds of data structures as well. It also
  helps generalize `MaybeOwned`, for similar reasons (see below.)

  A *very important* consequence is that the map methods using `Borrow` can
  potentially be put into a common `Map` trait that's implemented by `HashMap`,
  `TreeMap`, and others. While we do not propose to do so now, we definitely
  want to do so later on.

* When using a `HashMap<String, T>`, all of the basic methods like `find`,
  `contains_key` and `insert` "just work", without forcing you to think about
  `&String` vs `&str`.

* We don't need separate `_equiv` variants of methods. (However, this could
  probably be addressed with
  [multidispatch](https://github.com/rust-lang/rfcs/pull/195) by providing a
  blanket `Equiv` implementation.)

On the other hand, this approach retains some of the downsides of `_equiv`:

* The signature for methods like `find` and `contains_key` is more complex than
  their current signatures. There are two counterpoints. First, over time the
  `Borrow` trait is likely to become a well-known concept, so the signature will
  not appear completely alien. Second, what is perhaps more important than the
  signature is that, when using `find` on `HashMap<String, T>`, various method
  arguments *just work* as expected.

* The API does not guarantee "coherence": the `Hash` and `Eq` (or `Ord`, for
  `TreeMap`) implementations for the owned and borrowed keys might differ,
  breaking key invariants of the data structure. This is already the case with
  `_equiv`.

The [Alternatives section](#variants-of-borrow) includes a variant of `Borrow`
that doesn't suffer from these downsides, but has some downsides of its own.

### Clone-on-write (`Cow`) pointers

A side-benefit of the `Borrow` trait is that we can give a more general version
of the `MaybeOwned` as a "clone-on-write" smart pointer:

```rust
/// A generalization of Clone.
trait FromBorrow<Sized? B>: Borrow<B> {
    fn from_borrow(b: &B) -> Self;
}

/// A clone-on-write smart pointer
pub enum Cow<'a, T, B> where T: FromBorrow<B> {
    Shared(&'a B),
    Owned(T)
}

impl<'a, T, B> Cow<'a, T, B> where T: FromBorrow<B> {
    pub fn new(shared: &'a B) -> Cow<'a, T, B> {
        Shared(shared)
    }

    pub fn new_owned(owned: T) -> Cow<'static, T, B> {
        Owned(owned)
    }

    pub fn is_owned(&self) -> bool {
        match *self {
            Owned(_) => true,
            Shared(_) => false
        }
    }

    pub fn to_owned_mut(&mut self) -> &mut T {
        match *self {
            Shared(shared) => {
                *self = Owned(FromBorrow::from_borrow(shared));
                self.to_owned_mut()
            }
            Owned(ref mut owned) => owned
        }
    }

    pub fn into_owned(self) -> T {
        match self {
            Shared(shared) => FromBorrow::from_borrow(shared),
            Owned(owned) => owned
        }
    }
}

impl<'a, T, B> Deref<B> for Cow<'a, T, B> where T: FromBorrow<B>  {
    fn deref(&self) -> &B {
        match *self {
            Shared(shared) => shared,
            Owned(ref owned) => owned.borrow()
        }
    }
}

impl<'a, T, B> DerefMut<B> for Cow<'a, T, B> where T: FromBorrow<B> {
    fn deref_mut(&mut self) -> &mut B {
        self.to_owned_mut().borrow_mut()
    }
}
```

The type `Cow<'a, String, str>` is roughly equivalent to today's `MaybeOwned<'a>`
(and `Cow<'a, Vec<T>, [T]>` to `MaybeOwnedVector<'a, T>`).

By implementing `Deref` and `DerefMut`, the `Cow` type acts as a smart pointer
-- but in particular, the `mut` variant actually *clones* if the pointed-to
value is not currently owned. Hence "clone on write".

One slight gotcha with the design is that `&mut str` is not very useful, while
`&mut String` is (since it allows extending the string, for example). On the
other hand, `Deref` and `DerefMut` must deref to the *same* underlying type, and
for `Deref` to not require cloning, it must yield a `&str` value.

Thus, the `Cow` pointer offers a separate `to_owned_mut` method that yields a
mutable reference to the *owned* version of the type.

Note that, by not using `into_owned`, the `Cow` pointer itself may be owned by
some other data structure (perhaps as part of a collection) and will internally
track whether an owned copy is available.

Altogether, this RFC proposes to introduce `Borrow` and `Cow` as above, and to
deprecate `MaybeOwned` and `MaybeOwnedVector`. The API changes for the
collections are discussed [below](#the-apis).

## `IntoIterator` (and `Iterable`)

As discussed in [earlier](#iterable), some form of an `Iterable` trait is
desirable for both expressiveness and ergonomics. Unfortunately, a full
treatment of `Iterable` requires HKT for similar reasons to
[the collection traits](#lack-of-iterator-methods). However, it's possible to
get some of the way there in a forwards-compatible fashion.

In particular, the following two traits work fine (with
[associated items](https://github.com/rust-lang/rfcs/pull/195)):

```rust
trait Iterator {
    type A;
    fn next(&mut self) -> Option<A>;
    ...
}

trait IntoIterator {
    type A;
    type I: Iterator<A = A>;

    fn into_iter(self) -> I;
}
```

Because `IntoIterator` consumes `self`, lifetimes are not an issue.

It's tempting to also define a trait like:

```rust
trait Iterable<'a> {
    type A;
    type I: Iterator<&'a A>;

    fn iter(&'a self) -> I;
}
```

(along the lines of those proposed by
[an earlier RFC](https://github.com/rust-lang/rfcs/pull/17)).

The problem with `Iterable` as defined above is that it's locked to a particular
lifetime up front. But in many cases, the needed lifetime is not even nameable
in advance:

```rust
fn iter_through_rc<I>(c: Rc<I>) where I: Iterable<?> {
    // the lifetime of the borrow is established here,
    // so cannot even be named in the function signature
    for x in c.iter() {
        // ...
    }
}
```

To make this kind of example work, you'd need to be able to say something like:

```rust
where <'a> I: Iterable<'a>
```

that is, that `I` implements `Iterable` for *every* lifetime `'a`. While such a
feature is feasible to add to `where` clauses, the HKT solution is undoubtedly
cleaner.

Fortunately, we can have our cake and eat it too. This RFC proposes the
`IntoIterator` trait above, together with the following blanket `impl`:

```rust
impl<I: Iterator> IntoIterator for I {
    type A = I::A;
    type I = I;
    fn into_iter(self) -> I {
        self
    }
}
```

which means that taking `IntoIterator` is strictly more flexible than taking
`Iterator`. Note that in other languages (like Java), iterators are *not*
iterable because the latter implies an unlimited number of iterations. But
because `IntoIterator` consumes `self`, it yields only a single iteration, so
all is good.

For individual collections, one can then implement `IntoIterator` on both the
collection and borrows of it:

```rust
impl<T> IntoIterator for Vec<T> {
    type A = T;
    type I = MoveItems<T>;
    fn into_iter(self) -> MoveItems<T> { ... }
}

impl<'a, T> IntoIterator for &'a Vec<T> {
    type A = &'a T;
    type I = Items<'a, T>;
    fn into_iter(self) -> Items<'a, T> { ... }
}

impl<'a, T> IntoIterator for &'a mut Vec<T> {
    type A = &'a mut T;
    type I = ItemsMut<'a, T>;
    fn into_iter(self) -> ItemsMut<'a, T> { ... }
}
```

If/when HKT is added later on, we can add an `Iterable` trait and a blanket
`impl` like the following:

```rust
// the HKT version
trait Iterable {
    type A;
    type I<'a>: Iterator<&'a A>;
    fn iter<'a>(&'a self) -> I<'a>;
}

impl<'a, C: Iterable> IntoIterator for &'a C {
    type A = &'a C::A;
    type I = C::I<'a>;
    fn into_iter(self) -> I {
        self.iter()
    }
}
```

This gives a clean migration path: once `Vec` implements `Iterable`, it can drop
the `IntoIterator` `impl`s for borrowed vectors, since they will be covered by
the blanket implementation. No code should break.

Likewise, if we add a feature like the "universal" `where` clause mentioned
above, it can be used to deal with embedded lifetimes as in the
`iter_through_rc` example; and if the HKT version of `Iterable` is later added,
thanks to the suggested blanket `impl` for `IntoIterator` that `where` clause
could be changed to use `Iterable` instead, again without breakage.

### Benefits of `IntoIterator`

What do we gain by incorporating `IntoIterator` today?

This RFC proposes that `for` loops should use `IntoIterator` rather than
`Iterator`. With the blanket `impl` of `IntoIterator` for any `Iterator`, this
is not a breaking change. However, given the `IntoIterator` `impl`s for `Vec`
above, we would be able to write:

```rust
let v: Vec<Foo> = ...

for x in &v { ... }     // iterate over &Foo
for x in &mut v { ... } // iterate over &mut Foo
for x in v { ... }      // iterate over Foo
```

Similarly, methods that currently take slices or iterators can be changed to
take `IntoIterator` instead, immediately becoming more general and more
ergonomic.

In general, `IntoIterator` will allow us to move toward more `Iterator`-centric
APIs today, in a way that's compatible with HKT tomorrow.

### Additional methods

Another typical desire for an `Iterable` trait is to offer defaulted versions of
methods that basically re-export iterator methods on containers (see
[the earlier RFC](https://github.com/rust-lang/rfcs/pull/17)). Usually these
methods would go through a reference iterator (i.e. the `iter` method) rather
than a moving iterator.

It is possible to add such methods using the design proposed above, but there
are some drawbacks. For example, should `Vec::map` produce an iterator, or a new
vector?  It would be possible to do the latter generically, but only with
HKT. (See
[this discussion](https://github.com/rust-lang/rfcs/pull/17#issuecomment-43817453).)

This RFC only proposes to add the following method via `IntoIterator`, as a
convenience for a common pattern:

```rust
trait IterCloned {
    type A;
    type I: Iterator<A>;
    fn iter_cloned(self) -> I;
}

impl<'a, T, I: IntoIterator> IterCloned for I where I::A = &'a T {
    type A = T;
    type I = ClonedItems<I>;
    fn into_iter(self) -> I { ... }
}
```

(The `iter_cloned` method will help reduce the number of method variants in
general for collections, as we will see below).

We will leave to later RFCs the incorporation of additional methods. Notice, in
particular, that such methods can wait until we introduce an `Iterable` trait
via HKT without breaking backwards compatibility.

## Minimizing variants: `ByNeed` and `Predicate` traits

There are several kinds of methods that, in their most general form take
closures, but for which convenience variants taking simpler data are common:

* *Taking values by need*. For example, consider the `unwrap_or` and
  `unwrap_or_else` methods in `Option`:

  ```rust
  fn unwrap_or(self, def: T) -> T
  fn unwrap_or_else(self, f: || -> T) -> T
  ```

  The `unwrap_or_else` method is the most general: it invokes the closure to
  compute a default value *only when `self` is `None`*. When the default value
  is expensive to compute, this by-need approach helps. But often the default
  value is cheap, and closures are somewhat annoying to write, so `unwrap_or`
  provides a convenience wrapper.

* *Taking predicates*. For example, a method like `contains` often shows up
  (inconsistently!) in two variants:

  ```rust
  fn contains(&self, elem: &T) -> bool; // where T: PartialEq
  fn contains_fn(&self, pred: |&T| -> bool) -> bool;
  ```

  Again, the `contains_fn` version is the more general, but it's convenient to
  provide a specialized variant when the element type can be compared for
  equality, to avoid writing explicit closures.

As it turns out, with
[multidispatch](https://github.com/rust-lang/rfcs/pull/195)) it is possible to
use a *trait* to express these variants through overloading:

```rust
trait ByNeed<T> {
    fn compute(self) -> T;
}

impl<T> ByNeed<T> for T {
    fn compute(self) -> T {
        self
    }
}

// Due to multidispatch, this impl does NOT overlap with the above one
impl<T> ByNeed<T> for || -> T {
    fn compute(self) -> T {
        self()
    }
}

impl<T> Option<T> {
    fn unwrap_or<U>(self, def: U) where U: ByNeed<T> { ... }
    ...
}
```

```rust
trait Predicate<T> {
    fn check(&self, &T) -> bool;
}

impl<T: Eq> Predicate<T> for &T {
    fn check(&self, t: &T) -> bool {
        *self == t
    }
}

impl<T> Predicate<T> for |&T| -> bool {
    fn check(&self, t: &T) -> bool {
        (*self)(t)
    }
}

impl<T> Vec<T> {
    fn contains<P>(&self, pred: P) where P: Predicate<T> { ... }
    ...
}
```

Since these two patterns are particularly common throughout `std`, this RFC
proposes adding both of the above traits, and using them to cut down on the
number of method variants.

In particular, some methods on string slices currently work with `CharEq`, which
is similar to `Predicate<char>`:

```rust
pub trait CharEq {
    fn matches(&mut self, char) -> bool;
    fn only_ascii(&self) -> bool;
}
```

The difference is the `only_ascii` method, which is used to optimize certain
operations when the predicate only holds for characters in the ASCII range.

To keep these optimizations intact while connecting to `Predicate`, this RFC
proposes the following restructuring of `CharEq`:

```rust
pub trait CharPredicate: Predicate<char> {
    fn only_ascii(&self) -> bool {
        false
    }
}
```

### Why not leverage unboxed closures?

A natural question is: why not use the traits for unboxed closures to achieve a
similar effect? For example, you could imagine writing a blanket `impl` for
`Fn(&T) -> bool` for any `T: PartialEq`, which would allow `PartialEq` values to
be used anywhere a predicate-like closure was requested.

The problem is that these blanket `impl`s will often conflict. In particular,
*any* type `T` could implement `Fn() -> T`, and that single blanket `impl` would
preclude any others (at least, assuming that unboxed closure traits treat the
argument and return types as associated (output) types).

In addition, the explicit use of traits like `Predicate` makes the intended
semantics more clear, and the overloading less surprising.

## The APIs

Now we'll delve into the detailed APIs for the various concrete
collections. These APIs will often be given in tabular form, grouping together
common APIs across multiple collections. When writing these function signatures:

* We will assume a type parameter `T` for `Vec`, `BinaryHeap`, `DList` and `RingBuf`;
we will also use this parameter for APIs on `String`, where it should be
understood as `char`.

* We will assume type parameters `K: Borrow` and `V` for `HashMap` and
`TreeMap`; for `TrieMap` and `SmallIntMap` the `K` is assumed to be `uint`

* We will assume a type parameter `K: Borrow` for `HashSet` and `TreeSet`; for
  `BitvSet` it is assumed to be `uint`.

We will begin by outlining the most widespread APIs in tables, making it easy to
compare names and signatures across different kinds of collections. Then we will
focus on some APIs specific to particular classes of collections -- e.g. sets
and maps.  Finally, we will briefly discuss APIs that are specific to a single
concrete collection.

### Construction

All of the collections should support a static function:

```rust
fn new() -> Self
```

that creates an empty version of the collection; the constructor may take
arguments needed to set up the collection, e.g. the capacity for `LruCache`.

Several collections also support separate constructors for providing capacities in
advance; these are discussed [below](#capacity-management).

#### The `FromIterator` trait

All of the collections should implement the `FromIterator` trait:

```rust
pub trait FromIterator {
    type A:
    fn from_iter<T>(T) -> Self where T: IntoIterator<A = A>;
}
```

Note that this varies from today's `FromIterator` by consuming an `IntoIterator`
rather than `Iterator`. As explained [above](#intoiterator-and-iterable), this
choice is strictly more general and will not break any existing code.

This constructor initializes the collection with the contents of the
iterator. For maps, the iterator is over key/value pairs, and the semantics is
equivalent to inserting those pairs in order; if keys are repeated, the last
value is the one left in the map.

### Insertion

The table below gives methods for inserting items into various concrete collections:

Operation | Collections
--------- | -----------
`fn push(&mut self, T)`         | `Vec`, `BinaryHeap`, `String`
`fn push_front(&mut self, T)`   | `DList`, `RingBuf`
`fn push_back(&mut self, T)`    | `DList`, `RingBuf`
`fn insert(&mut self, uint, T)` | `Vec`, `RingBuf`, `String`
`fn insert(&mut self, K::Owned) -> bool` | `HashSet`, `TreeSet`, `TrieSet`, `BitvSet`
`fn insert(&mut self, K::Owned, V) -> Option<V>`    | `HashMap`, `TreeMap`, `TrieMap`, `SmallIntMap`
`fn append(&mut self, Self)`        | `DList`
`fn prepend(&mut self, Self)`        | `DList`

There are a few changes here from the current state of affairs:

* The `DList` and `RingBuf` data structures no longer provide `push`, but rather
  `push_front` and `push_back`. This change is based on (1) viewing them as
  deques and (2) not giving priority to the "front" or the "back".

* The `insert` method on maps returns the value previously associated with the
  key, if any. Previously, this functionality was provided by a `swap` method,
  which has been dropped (consolidating needless method variants.)

Aside from these changes, a number of insertion methods will be deprecated
(e.g. the `append` and `append_one` methods on `Vec`). These are discussed
further in the section on "specialized operations"
[below](#specialized-operations).

#### The `Extend` trait (was: `Extendable`)

In addition to the standard insertion operations above, *all* collections will
implement the `Extend` trait. This trait was previously called `Extendable`, but
in general we
[prefer to avoid](http://aturon.github.io/style/naming/README.html) `-able`
suffixes and instead name the trait using a verb (or, especially, the key method
offered by the trait.)

The `Extend` trait allows data from an arbitrary iterator to be inserted into a
collection, and will be defined as follows:

```rust
pub trait Extend: FromIterator {
    fn extend<T>(&mut self, T) where T: IntoIterator<A = Self::A>;
}
```

As with `FromIterator`, this trait has been modified to take an `IntoIterator`
value.

### Deletion

The table below gives methods for removing items into various concrete collections:

Operation | Collections
--------- | -----------
`fn clear(&mut self)` | *all*
`fn pop(&mut self) -> Option<T>` | `Vec`, `BinaryHeap`, `String`
`fn pop_front(&mut self) -> Option<T>` | `DList`, `RingBuf`
`fn pop_back(&mut self) -> Option<T>` | `DList`, `RingBuf`
`fn remove(&mut self, uint) -> Option<T>` | `Vec`, `RingBuf`, `String`
`fn remove(&mut self, &K) -> bool` | `HashSet`, `TreeSet`, `TrieSet`, `BitvSet`
`fn remove(&mut self, &K) -> Option<V>` | `HashMap`, `TreeMap`, `TrieMap`, `SmallIntMap`
`fn truncate(&mut self, len: uint)` | `Vec`, `String`, `Bitv`, `DList`, `RingBuf`
`fn retain<P>(&mut self, f: P) where P: Predicate<T>` | `Vec`, `DList`, `RingBuf`
`fn dedup(&mut self)` | `Vec`, `DList`, `RingBuf` where `T: PartialEq`

As with the insertion methods, there are some differences from today's API:

* The `DList` and `RingBuf` data structures no longer provide `pop`, but rather
  `pop_front` and `pop_back` -- similarly to the `push` methods.

* The `remove` method on maps returns the value previously associated with the
  key, if any. Previously, this functionality was provided by a separate `pop`
  method, which has been dropped (consolidating needless method variants.)

* The `retain` method takes a `Predicate`.

* The `truncate`, `retain` and `dedup` methods are offered more widely.

Again, some of the more specialized methods are not discussed here; see
"specialized operations" [below](#specialized-operations).

### Inspection/mutation

The next table gives methods for inspection and mutation of existing items in collections:

Operation | Collections
--------- | -----------
`fn len(&self) -> uint` | *all*
`fn is_empty(&self) -> bool` | *all*
`fn get(&self, uint) -> Option<&T>` | `[T]`, `Vec`, `RingBuf`
`fn get_mut(&mut self, uint) -> Option<&mut T>` | `[T]`, `Vec`, `RingBuf`
`fn get(&self, &K) -> Option<&V>` | `HashMap`, `TreeMap`, `TrieMap`, `SmallIntMap`
`fn get_mut(&mut self, &K) -> Option<&mut V>` | `HashMap`, `TreeMap`, `TrieMap`, `SmallIntMap`
`fn contains<P>(&self, P) where P: Predicate<T>` | `[T]`, `str`, `Vec`, `String`, `DList`, `RingBuf`, `BinaryHeap`
`fn contains(&self, &K) -> bool` | `HashSet`, `TreeSet`, `TrieSet`, `EnumSet`
`fn contains_key(&self, &K) -> bool` | `HashMap`, `TreeMap`, `TrieMap`, `SmallIntMap`

The biggest changes from the current APIs are:

* The `find` and `find_mut` methods have been renamed to `get` and `get_mut`.
  Further, all `get` methods return `Option` values and do not invoke `fail!`.
  This is part of a general convention described in the next section (on the
  `Index` traits).

* The `contains` method is offered more widely.

* There is no longer an equivalent of `find_copy` (which should be called
  `find_clone`). Instead, we propose to add the following method to the `Option<&'a T>`
  type where `T: Clone`:

  ```rust
  fn cloned(self) -> Option<T> {
      self.map(|x| x.clone())
  }
  ```

  so that `some_map.find_copy(key)` will instead be written
  `some_map.find(key).cloned()`. This method chain is slightly longer, but is
  more clear and allows us to drop the `_copy` variants. Moreover, *all* users
  of `Option` benefit from the new convenience method.

#### The `Index` trait

The `Index` and `IndexMut` traits provide indexing notation like `v[0]`:

```rust
pub trait Index {
    type Index;
    type Result;
    fn index(&'a self, index: &Index) -> &'a Result;
}

pub trait IndexMut {
    type Index;
    type Result;
    fn index_mut(&'a mut self, index: &Index) -> &'a mut Result;
}
```

These traits will be implemented for: `[T]`, `Vec`, `RingBuf`, `HashMap`, `TreeMap`, `TrieMap`, `SmallIntMap`.

As a general convention, implementation of the `Index` traits will *fail the
task* if the index is invalid (out of bounds or key not found); they will
therefor return direct references to values. Any collection implementing `Index`
(resp. `IndexMut`) should also provide a `get` method (resp. `get_mut`) as a
non-failing variant that returns an `Option` value.

This allows us to keep indexing notation maximally concise, while still
providing convenient non-failing variants (which can be used to provide a check
for index validity).

### Iteration

Every collection should provide the standard trio of iteration methods:

```rust
fn iter(&'a self) -> Items<'a>;
fn iter_mut(&'a mut self) -> ItemsMut<'a>;
fn into_iter(self) -> ItemsMove;
```

and in particular implement the `IntoIterator` trait on both the collection type
and on (mutable) references to it.

### Capacity management

many of the collections have some notion of "capacity", which may be fixed, grow
explicitly, or grow implicitly:

- No capacity/fixed capacity: `DList`, `TreeMap`, `TreeSet`, `TrieMap`, `TrieSet`, slices, `EnumSet`
- Explicit growth: `LruCache`
- Implicit growth: `Vec`, `RingBuf`, `HashMap`, `HashSet`, `BitvSet`, `BinaryHeap`

Growable collections provide functions for capacity management, as follows.

#### Explicit growth

For explicitly-grown collections, the normal constructor (`new`) takes a
capacity argument. Capacity can later be inspected or updated as follows:

```rust
fn capacity(&self) -> uint
fn set_capacity(&mut self, capacity: uint)
```

(Note, this renames `LruCache::change_capacity` to `set_capacity`, the
prevailing style for setter method.)

#### Implicit growth

For implicitly-grown collections, the normal constructor (`new`) does not take a
capacity, but there is an explicit `with_capacity` constructor, along with other
functions to work with the capacity later on:

```rust
fn with_capacity(uint) -> Self
fn capacity(&self) -> uint
fn reserve(&mut self, additional: uint)
fn reserve_exact(&mut self, additional: uint)
fn shrink_to_fit(&mut self)
```

There are some important changes from the current APIs:

* The `reserve` and `reserve_exact` methods now take as an argument the *extra*
  space to reserve, rather than the final desired capacity, as this usage is
  vastly more common. The `reserve` function may grow the capacity by a larger
  amount than requested, to ensure amortization, while `reserve_exact` will
  reserve exactly the requested additional capacity. The `reserve_additional`
  methods are deprecated.

* The `with_capacity` constructor does *not* take any additional arguments, for
  uniformity with `new`. This change affects `Bitv` in particular.

#### Bounded iterators

Some of the maps (e.g. `TreeMap`) currently offer specialized iterators over
their entries starting at a given key (called `lower_bound`) and above a given
key (called `upper_bound`), along with `_mut` variants. While the functionality
is worthwhile, the names are not very clear, so this RFC proposes the following
replacement API (thanks to [@Gankro for the suggestion](https://github.com/rust-lang/rfcs/pull/235#issuecomment-55512788)):

```rust
Bound<T> {
    /// An inclusive bound
    Included(T),

    /// An exclusive bound
    Excluded(T),

    Unbounded,
}

/// Creates a double-ended iterator over a sub-range of the collection's items,
/// starting at min, and ending at max. If min is `Unbounded`, then it will
/// be treated as "negative infinity", and if max is `Unbounded`, then it will
/// be treated as "positive infinity". Thus range(Unbounded, Unbounded) will yield
/// the whole collection.
fn range(&self, min: Bound<&T>, max: Bound<&T>) -> RangedItems<'a, T>;

fn range_mut(&self, min: Bound<&T>, max: Bound<&T>) -> RangedItemsMut<'a, T>;
```

These iterators should be provided for any maps over ordered keys (`TreeMap`,
`TrieMap` and `SmallIntMap`).

In addition, analogous methods should be provided for sets over ordered keys
(`TreeSet`, `TrieSet`, `BitvSet`).

### Set operations

#### Comparisons

All sets should offer the following methods, as they do today:

```rust
fn is_disjoint(&self, other: &Self) -> bool;
fn is_subset(&self, other: &Self) -> bool;
fn is_superset(&self, other: &Self) -> bool;
```

#### Combinations

Sets can also be combined using the standard operations -- union, intersection,
difference and symmetric difference (exclusive or). Today's APIs for doing so
look like this:

```rust
fn union<'a>(&'a self, other: &'a Self) -> I;
fn intersection<'a>(&'a self, other: &'a Self) -> I;
fn difference<'a>(&'a self, other: &'a Self) -> I;
fn symmetric_difference<'a>(&'a self, other: &'a Self) -> I;
```

where the `I` type is an iterator over keys that varies by concrete
set. Working with these iterators avoids materializing intermediate
sets when they're not needed; the `collect` method can be used to
create sets when they are. This RFC proposes to keep these names
intact, following the
[RFC](https://github.com/rust-lang/rfcs/pull/344) on iterator
conventions.

Sets should also implement the `BitOr`, `BitAnd`, `BitXor` and `Sub` traits from
`std::ops`, allowing overloaded notation `|`, `&`, `|^` and `-` to be used with
sets. These are equivalent to invoking the corresponding `iter_` method and then
calling `collect`, but for some sets (notably `BitvSet`) a more efficient direct
implementation is possible.

Unfortunately, we do not yet have a set of traits corresponding to operations
`|=`, `&=`, etc, but again in some cases doing the update in place may be more
efficient. Right now, `BitvSet` is the only concrete set offering such operations:

```rust
fn union_with(&mut self, other: &BitvSet)
fn intersect_with(&mut self, other: &BitvSet)
fn difference_with(&mut self, other: &BitvSet)
fn symmetric_difference_with(&mut self, other: &BitvSet)
```

This RFC punts on the question of naming here: it does *not* propose a new set
of names. Ideally, we would add operations like `|=` in a separate RFC, and use
those conventionally for sets. If not, we will choose fallback names during the
stabilization of `BitvSet`.

### Map operations

#### Combined methods

The `HashMap` type currently provides a somewhat bewildering set of `find`/`insert` variants:

```rust
fn find_or_insert(&mut self, k: K, v: V) -> &mut V
fn find_or_insert_with<'a>(&'a mut self, k: K, f: |&K| -> V) -> &'a mut V
fn insert_or_update_with<'a>(&'a mut self, k: K, v: V, f: |&K, &mut V|) -> &'a mut V
fn find_with_or_insert_with<'a, A>(&'a mut self, k: K, a: A, found: |&K, &mut V, A|, not_found: |&K, A| -> V) -> &'a mut V
```

These methods are used to couple together lookup and insertion/update
operations, thereby avoiding an extra lookup step. However, the current set of
method variants seems overly complex.

There is [another RFC](https://github.com/rust-lang/rfcs/pull/216) already in
the queue addressing this problem in a very nice way, and this RFC defers to
that one

#### Key and value iterators

In addition to the standard iterators, maps should provide by-reference
convenience iterators over keys and values:

```rust
fn keys(&'a self) -> Keys<'a, K>
fn values(&'a self) -> Values<'a, V>
```

While these iterators are easy to define in terms of the main `iter` method,
they are used often enough to warrant including convenience methods.

### Specialized operations

Many concrete collections offer specialized operations beyond the ones given
above. These will largely be addressed through the API stabilization process
(which focuses on local API issues, as opposed to general conventions), but a
few broad points are addressed below.

#### Relating `Vec` and `String` to slices

One goal of this RFC is to supply all of the methods on (mutable) slices on
`Vec` and `String`. There are a few ways to achieve this, so concretely the
proposal is for `Vec<T>` to implement `Deref<[T]>` and `DerefMut<[T]>`, and
`String` to implement `Deref<str>`. This will automatically allow all slice
methods to be invoked from vectors and strings, and will allow writing `&*v`
rather than `v.as_slice()`.

In this scheme, `Vec` and `String` are really "smart pointers" around the
corresponding slice types. While counterintuitive at first, this perspective
actually makes a fair amount of sense, especially with DST.

(Initially, it was unclear whether this strategy would play well with method
resolution, but the planned resolution rules should work fine.)

#### `String` API

One of the key difficulties with the `String` API is that strings use utf8
encoding, and some operations are only efficient when working at the byte level
(and thus taking this encoding into account).

As a general principle, we will move the API toward the following convention:
index-related operations always work in terms of bytes, other operations deal
with chars by default (but can have suffixed variants for working at other
granularities when appropriate.)

#### `DList`

The `DList` type offers a number of specialized methods:

```rust
swap_remove, insert_when, insert_ordered, merge, rotate_forward and rotate_backward
```

Prior to stabilizing the `DList` API, we will attempt to simplify its API
surface, possibly by using idea from the
[collection views RFC](https://github.com/rust-lang/rfcs/pull/216).

### Minimizing method variants via iterators

#### Partitioning via `FromIterator`

One place we can move toward iterators is functions like `partition` and
`partitioned` on vectors and slices:

```rust
// on Vec<T>
fn partition(self, f: |&T| -> bool) -> (Vec<T>, Vec<T>);

// on [T] where T: Clone
fn partitioned(&self, f: |&T| -> bool) -> (Vec<T>, Vec<T>);
```

These two functions transform a vector/slice into a pair of vectors, based on a
"partitioning" function that says which of the two vectors to place elements
into. The `partition` variant works by moving elements of the vector, while
`paritioned` clones elements.

There are a few unfortunate aspects of an API like this one:

* It's specific to vectors/slices, although in principle both the source and
  target containers could be more general.

* The fact that two variants have to be exposed, for owned versus clones, is
  somewhat unfortunate.

This RFC proposes the following alternative design:

```rust
pub enum Either<T, U> {
    pub Left(T),
    pub Right(U),
}

impl<A, B> FromIterator for (A, B) where A: Extend, B: Extend {
    fn from_iter<I>(mut iter: I) -> (A, B) where I: IntoIterator<Either<T, U>> {
        let mut left: A = FromIterator::from_iter(None::<T>);
        let mut right: B = FromIterator::from_iter(None::<U>);

        for item in iter {
            match item {
                Left(t) => left.extend(Some(t)),
                Right(u) => right.extend(Some(u)),
            }
        }

        (left, right)
    }
}

trait Iterator {
    ...
    fn partition(self, |&A| -> bool) -> Partitioned<A> { ... }
}

// where Partitioned<A>: Iterator<A = Either<A, A>>
```

This design drastically generalizes the partitioning functionality, allowing it
be used with arbitrary collections and iterators, while removing the
by-reference and by-value distinction.

Using this design, you have:

```rust
// The following two lines are equivalent:
let (u, w) = v.partition(f);
let (u, w): (Vec<T>, Vec<T>) = v.into_iter().partition(f).collect();

// The following two lines are equivalent:
let (u, w) = v.as_slice().partitioned(f);
let (u, w): (Vec<T>, Vec<T>) = v.iter_cloned().partition(f).collect();
```

There is some extra verbosity, mainly due to the type annotations for `collect`,
but the API is much more flexible, since the partitioned data can now be
collected into other collections (or even differing collections). In addition,
partitioning is supported for *any* iterator.

#### Removing methods like `from_elem`, `from_fn`, `grow`, and `grow_fn`

Vectors and some other collections offer constructors and growth functions like
the following:

```rust
fn from_elem(length: uint, value: T) -> Vec<T>
fn from_fn(length: uint, op: |uint| -> T) -> Vec<T>
fn grow(&mut self, n: uint, value: &T)
fn grow_fn(&mut self, n: uint, f: |uint| -> T)
```

These extra variants can easily be dropped in favor of iterators, and this RFC
proposes to do so.

The `iter` module already contains a `Repeat` iterator; this RFC proposes to add
a free function `repeat` to `iter` as a convenience for `iter::Repeat::new`.

With that in place, we have:

```rust
// Equivalent:
let v = Vec::from_elem(n, a);
let v = Vec::from_iter(repeat(a).take(n));

// Equivalent:
let v = Vec::from_fn(n, f);
let v = Vec::from_iter(range(0, n).map(f));

// Equivalent:
v.grow(n, a);
v.extend(repeat(a).take(n));

// Equivalent:
v.grow_fn(n, f);
v.extend(range(0, n).map(f));
```

While these replacements are slightly longer, an important aspect of ergonomics
is *memorability*: by placing greater emphasis on iterators, programmers will
quickly learn the iterator APIs and have those at their fingertips, while
remembering ad hoc method variants like `grow_fn` is more difficult.

#### Long-term: removing `push_all` and `push_all_move`

The `push_all` and `push_all_move` methods on vectors are yet more API variants
that could, in principle, go through iterators:

```rust
// The following are *semantically* equivalent
v.push_all(some_slice);
v.extend(some_slice.iter_cloned());

// The following are *semantically* equivalent
v.push_all_move(some_vec);
v.extend(some_vec);
```

However, currently the `push_all` and `push_all_move` methods can rely
on the *exact* size of the container being pushed, in order to elide
bounds checks. We do not currently have a way to "trust" methods like
`len` on iterators to elide bounds checks. A separate RFC will
introduce the notion of a "trusted" method which should support such
optimization and allow us to deprecate the `push_all` and
`push_all_move` variants. (This is unlikely to happen before 1.0, so
the methods will probably still be included with "experimental"
status, and likely with different names.)

# Alternatives

## `Borrow` and the `Equiv` problem

### Variants of `Borrow`

The original version of `Borrow` was somewhat more subtle:

```rust
/// A trait for borrowing.
/// If `T: Borrow` then `&T` represents data borrowed from `T::Owned`.
trait Borrow for Sized? {
    /// The type being borrowed from.
    type Owned;

    /// Immutably borrow from an owned value.
    fn borrow(&Owned) -> &Self;

    /// Mutably borrow from an owned value.
    fn borrow_mut(&mut Owned) -> &mut Self;
}

trait ToOwned: Borrow {
    /// Produce a new owned value, usually by cloning.
    fn to_owned(&self) -> Owned;
}

impl<A: Sized> Borrow for A {
    type Owned = A;
    fn borrow(a: &A) -> &A {
        a
    }
    fn borrow_mut(a: &mut A) -> &mut A {
        a
    }
}

impl<A: Clone> ToOwned for A {
    fn to_owned(&self) -> A {
        self.clone()
    }
}

impl Borrow for str {
    type Owned = String;
    fn borrow(s: &String) -> &str {
        s.as_slice()
    }
    fn borrow_mut(s: &mut String) -> &mut str {
        s.as_mut_slice()
    }
}

impl ToOwned for str {
    fn to_owned(&self) -> String {
        self.to_string()
    }
}

impl<T> Borrow for [T] {
    type Owned = Vec<T>;
    fn borrow(s: &Vec<T>) -> &[T] {
        s.as_slice()
    }
    fn borrow_mut(s: &mut Vec<T>) -> &mut [T] {
        s.as_mut_slice()
    }
}

impl<T> ToOwned for [T] {
    fn to_owned(&self) -> Vec<T> {
        self.to_vec()
    }
}

impl<K,V> HashMap<K,V> where K: Borrow + Hash + Eq {
    fn find(&self, k: &K) -> &V { ... }
    fn insert(&mut self, k: K::Owned, v: V) -> Option<V> { ... }
    ...
}

pub enum Cow<'a, T> where T: ToOwned {
    Shared(&'a T),
    Owned(T::Owned)
}
```

This approach ties `Borrow` directly to the borrowed data, and uses an
associated type to *uniquely determine* the corresponding owned data type.

For string keys, we would use `HashMap<str, V>`. Then, the `find` method would
take an `&str` key argument, while `insert` would take an owned `String`. On the
other hand, for some other type `Foo` a `HashMap<Foo, V>` would take
`&Foo` for `find` and `Foo` for `insert`. (More discussion on the choice of
ownership is given in the [alternatives section](#ownership-management-for-keys).

**Benefits of this alternative**:

* Unlike the current `_equiv` or `find_with` methods, or the proposal in the
RFC, this approach guarantees coherence about hashing or ordering. For example,
`HashMap` above requires that `K` (the borrowed key type) is `Hash`, and will
produce hashes from owned keys by first borrowing from them.

* Unlike the proposal in this RFC, the signature of the methods for maps is
  *very simple* -- essentially the same as the current `find`, `insert`, etc.

* Like the proposal in this RFC, there is only a single `Borrow`
  trait, so it would be possible to standardize on a `Map` trait later
  on and include these APIs. The trait could be made somewhat simpler
  with this alternative form of `Borrow`, but can be provided in
  either case; see
  [these](https://github.com/rust-lang/rfcs/pull/235#issuecomment-55976755)
  [comments](https://github.com/rust-lang/rfcs/pull/235#issuecomment-56070223)
  for details.

* The `Cow` data type is simpler than in the RFC's proposal, since it does not
  need a type parameter for the owned data.

**Drawbacks of this alternative**:

* It's quite subtle that you want to use `HashMap<str, T>` rather than
  `HashMap<String, T>`. That is, if you try to use a map in the "obvious way"
  you will not be able to use string slices for lookup, which is part of what
  this RFC is trying to achieve. The same applies to `Cow`.

* The design is somewhat less flexible than the one in the RFC, because (1)
  there is a fixed choice of owned type corresponding to each borrowed type and
  (2) you cannot use multiple borrow types for lookups at different types
  (e.g. using `&String` sometimes and `&str` other times). On the other hand,
  these restrictions guarantee coherence of hashing/equality/comparison.

* This version of `Borrow`, mapping from borrowed to owned data, is
  somewhat less intuitive.

On the balance, the approach proposed in the RFC seems better, because using the
map APIs in the obvious ways works by default.

### The `HashMapKey` trait and friends

An earlier proposal for solving the `_equiv` problem was given in the
[associated items RFC](https://github.com/rust-lang/rfcs/pull/195)):

```rust
trait HashMapKey : Clone + Hash + Eq {
    type Query: Hash = Self;
    fn compare(&self, other: &Query) -> bool { self == other }
    fn query_to_key(q: &Query) -> Self { q.clone() };
}

impl HashMapKey for String {
    type Query = str;
    fn compare(&self, other: &str) -> bool {
        self.as_slice() == other
    }
    fn query_to_key(q: &str) -> String {
        q.into_string()
    }
}

impl<K,V> HashMap<K,V> where K: HashMapKey {
    fn find(&self, q: &K::Query) -> &V { ... }
}
```

This solution has several drawbacks, however:

* It requires a separate trait for different kinds of maps -- one for `HashMap`,
  one for `TreeMap`, etc.

* It requires that a trait be implemented on a given key without providing a
  blanket implementation. Since you also need different traits for different
  maps, it's easy to imagine cases where a out-of-crate type you want to use as
  a key doesn't implement the key trait, forcing you to newtype.

* It doesn't help with the `MaybeOwned` problem.

### Daniel Micay's hack

@strcat has a [PR](https://github.com/rust-lang/rust/pull/16713) that makes it
possible to, for example, coerce a `&str` to an `&String` value.

This provides some help for the `_equiv` problem, since the `_equiv` methods
could potentially be dropped. However, there are a few downsides:

* Using a map with string keys is still a bit more verbose:

  ```rust
  map.find("some static string".as_string()) // with the hack
  map.find("some static string")             // with this RFC
  ```

* The solution is specialized to strings and vectors, and does not necessarily
  support user-defined unsized types or slices.

* It doesn't help with the `MaybeOwned` problem.

* It exposes some representation interplay between slices and references to
  owned values, which we may not want to commit to or reveal.

## For `IntoIterator`

### Handling of `for` loops

The fact that `for x in v` moves elements from `v`, while `for x in v.iter()`
yields references, may be a bit surprising. On the other hand, moving is the
default almost everywhere in Rust, and with the proposed approach you get to use `&` and
`&mut` to easily select other forms of iteration.

(See
[@huon's comment](https://github.com/rust-lang/rfcs/pull/235/files#r17697796)
for additional drawbacks.)

Unfortunately, it's a bit tricky to make for use by-ref iterators instead. The
problem is that an iterator is `IntoIterator`, but it is not `Iterable` (or
whatever we call the by-reference trait). Why? Because `IntoIterator` gives you
an iterator that can be used only *once*, while `Iterable` allows you to ask for
iterators repeatedly.

If `for` demanded an `Iterable`, then `for x in v.iter()` and `for x in v.iter_mut()`
would cease to work -- we'd have to find some other approach. It might be
doable, but it's not obvious how to do it.

### Input versus output type parameters

An important aspect of the `IntoIterator` design is that the element type is an
associated type, *not* an input type.

This is a tradeoff:

* Making it an associated type means that the `for` examples work, because the
  type of `Self` uniquely determines the element type for iteration, aiding type
  inference.

* Making it an input type would forgo those benefits, but would allow some
  additional flexibility. For example, you could implement `IntoIterator<A>` for
  an iterator on `&A` when `A` is cloned, therefore *implicitly* cloning as
  needed to make the ownership work out (and obviating the need for
  `iter_cloned`). However, we have generally kept away from this kind of
  implicit magic, *especially* when it can involve hidden costs like cloning, so
  the more explicit design given in this RFC seems best.

# Downsides

Design tradeoffs were discussed inline.

# Unresolved questions

## Unresolved conventions/APIs

As mentioned [above](#combinations), this RFC does not resolve the question of
what to call set operations that update the set in place.

It likewise does not settle the APIs that appear in only single concrete
collections. These will largely be handled through the API stabilization
process, unless radical changes are proposed.

Finally, additional methods provided via the `IntoIterator` API are left for
future consideration.

## Coercions

Using the `Borrow` trait, it might be possible to safely add a coercion for auto-slicing:

```
  If T: Borrow:
    coerce  &'a T::Owned      to  &'a T
    coerce  &'a mut T::Owned  to  &'a mut T
```

For sized types, this coercion is *forced* to be trivial, so the only time it
would involve running user code is for unsized values.

A general story about such coercions will be left to a
[follow-up RFC](https://github.com/rust-lang/rfcs/pull/241).
