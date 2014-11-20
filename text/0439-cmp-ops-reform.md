- Start Date: 2014-11-03
- RFC PR: [rust-lang/rfcs#439](https://github.com/rust-lang/rfcs/pull/439)
- Rust Issue: [rust-lang/rfcs#19148](https://github.com/rust-lang/rust/issues/19148)

# Summary

This RFC proposes a number of design improvements to the `cmp` and
`ops` modules in preparation for 1.0. The impetus for these
improvements, besides the need for stabilization, is that we've added
several important language features (like multidispatch) that greatly
impact the design. Highlights:

* Make basic unary and binary operators work by value and use associated types.
* Generalize comparison operators to work across different types; drop `Equiv`.
* Refactor slice notation in favor of *range notation* so that special
  traits are no longer needed.
* Add `IndexSet` to better support maps.
* Clarify ownership semantics throughout.

# Motivation

The operator and comparison traits play a double role: they are lang
items known to the compiler, but are also library APIs that need to be
stabilized.

While the traits have been fairly stable, a lot has changed in the
language recently, including the addition of multidispatch, associated
types, and changes to method resolution (especially around smart
pointers). These are all things that impact the ideal design of the traits.

Since it is now relatively clear how these language features will work
at 1.0, there is enough information to make final decisions about the
construction of the comparison and operator traits. That's what this
RFC aims to do.

# Detailed design

The traits in `cmp` and `ops` can be broken down into several
categories, and to keep things manageable this RFC discusses each
category separately:

* Basic operators:
  * Unary: `Neg`, `Not`
  * Binary: `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Shl`, `Shr`, `BitAnd`, `BitOr`, `BitXor`,
* Comparison: `PartialEq`, `PartialOrd`, `Eq`, `Ord`, `Equiv`
* Indexing and slicing: `Index`, `IndexMut`, `Slice`, `SliceMut`
* Special traits: `Deref`, `DerefMut`, `Drop`, `Fn`, `FnMut`, `FnOnce`

## Basic operators

The basic operators include arithmetic and bitwise notation with both
unary and binary operators.

### Current design

Here are two example traits, one unary and one binary, for basic operators:

```rust
pub trait Not<Result> {
    fn not(&self) -> Result;
}

pub trait Add<Rhs, Result> {
    fn add(&self, rhs: &Rhs) -> Result;
}
```

The rest of the operators follow the same pattern. Note that `self`
and `rhs` are taken by reference, and the compiler introduce *silent*
uses of `&` for the operands.

The traits also take `Result` as an
[*input*](https://github.com/rust-lang/rfcs/pull/195) type.

### Proposed design

This RFC proposes to make `Result` an associated (output) type, and to
make the traits work by value:

```rust
pub trait Not {
    type Result;
    fn not(self) -> Result;
}

pub trait Add<Rhs = Self> {
    type Result;
    fn add(self, rhs: Rhs) -> Result;
}
```

The reason to make `Result` an associated type is straightforward: it
should be uniquely determined given `Self` and other input types, and
making it an associated type is better for both type inference and for
keeping things concise when using these traits in bounds.

Making these traits work by value is motivated by cases like `DList`
concatenation, where you may want the operator to actually consume the
operands in producing its output (by welding the two lists together).

It also means that the compiler does not have to introduce a silent
`&` for the operands, which means that the ownership semantics when
using these operators is much more clear.

Fortunately, there is no loss in expressiveness, since you can always
implement the trait on reference types. However, for types that *do*
need to be taken by reference, there is a slight loss in ergonomics
since you may need to explicitly borrow the operands with `&`. The
upside is that the ownership semantics become clearer: they more
closely resemble normal function arguments.

By keeping `Rhs` as an input trait on the trait, you can overload on the
types of both operands via
[multidispatch](https://github.com/rust-lang/rfcs/pull/195).  By
defaulting `Rhs` to `Self`, in
[the future](https://github.com/rust-lang/rfcs/pull/213) it will be
possible to simply say `T: Add` as shorthand for `T: Add<T>`, which is
the common case.

Examples:

```rust
// Basic setup for Copy types:
impl Add<uint> for uint {
    type Result = uint;
    fn add(self, rhs: uint) -> uint { ... }
}

// Overloading on the Rhs:
impl Add<uint> for Complex {
    type Result = Complex;
    fn add(self, rhs: uint) -> Complex { ... }
}

impl Add<Complex> for Complex {
    type Result = Complex;
    fn add(self, rhs: Complex) -> Complex { ... }
}

// Recovering by-ref semantics:
impl<'a, 'b> Add<&'a str> for &'b str {
    type Result = String;
    fn add(self, rhs: &'a str) -> String { ... }
}
```

## Comparison traits

The comparison traits provide overloads for operators like `==` and `>`.

### Current design

Comparisons are subtle, because some types (notably `f32` and `f64`)
do not actually provide full equivalence relations or total
orderings. The current design therefore splits the comparison traits
into "partial" variants that do not promise full equivalence
relations/ordering, and "total" variants which inherit from them but
make stronger semantic guarantees. The floating point types implement
the partial variants, and the operators defer to them. But certain
collection types require e.g. total rather than partial orderings:

```rust
pub trait PartialEq {
    fn eq(&self, other: &Self) -> bool;

    fn ne(&self, other: &Self) -> bool { !self.eq(other) }
}

pub trait Eq: PartialEq {}

pub trait PartialOrd: PartialEq {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>;
    fn lt(&self, other: &Self) -> bool { .. }
    fn le(&self, other: &Self) -> bool { .. }
    fn gt(&self, other: &Self) -> bool { .. }
    fn ge(&self, other: &Self) -> bool { .. }
}

pub trait Ord: Eq + PartialOrd {
    fn cmp(&self, other: &Self) -> Ordering;
}

pub trait Equiv<T> {
    fn equiv(&self, other: &T) -> bool;
}
```

In addition there is an `Equiv` trait that can be used to compare
values of *different* types for equality, but does not correspond to
any operator sugar. (It was introduced in part to help solve some
problems in map APIs, which are now resolved in a different way.)

The comparison traits all work by reference, and the compiler inserts
implicit uses of `&` to make this ergonomic.

### Proposed design

This RFC proposes to follow largely the same design strategy, but to
remove `Equiv` and instead generalize the traits via multidispatch:

```rust
pub trait PartialEq<Rhs = Self> {
    fn eq(&self, other: &Rhs) -> bool;

    fn ne(&self, other: &Rhs) -> bool { !self.eq(other) }
}

pub trait Eq<Rhs = Self>: PartialEq<Rhs> {}

pub trait PartialOrd<Rhs = Self>: PartialEq<Rhs> {
    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;
    fn lt(&self, other: &Rhs) -> bool { .. }
    fn le(&self, other: &Rhs) -> bool { .. }
    fn gt(&self, other: &Rhs) -> bool { .. }
    fn ge(&self, other: &Rhs) -> bool { .. }
}

pub trait Ord<Rhs = Self>: Eq<Rhs> + PartialOrd<Rhs> {
    fn cmp(&self, other: &Rhs) -> Ordering;
}
```

Due to the use of defaulting, this generalization loses no
ergonomics. However, it makes it *possible* to overload notation like
`==` to compare different types without needing an explicit
conversion. (Precisely *which* overloadings we provide in `std` will
be subject to API stabilization.) This more general design will allow
us to eliminate the `iter::order` submodule in favor of comparison
notation, for example.

This design suffers from the problem that it is somewhat painful to
implement or derive `Eq`/`Ord`, which is the common case. We can
likely improve e.g. `#[deriving(Ord)]` to automatically derive
`PartialOrd`. See Alternatives for a more radical design (and the
reasons that it's not feasible right now.)

## Indexing and slicing

There are a few traits that support `[]` notation for indexing and slicing.

### Current design:

The current design is as follows:

```rust
pub trait Index<Index, Sized? Result> {
    fn index<'a>(&'a self, index: &Index) -> &'a Result;
}

pub trait IndexMut<Index, Result> {
    fn index_mut<'a>(&'a mut self, index: &Index) -> &'a mut Result;
}

pub trait Slice<Idx, Sized? Result> for Sized? {
    fn as_slice_<'a>(&'a self) -> &'a Result;
    fn slice_from_or_fail<'a>(&'a self, from: &Idx) -> &'a Result;
    fn slice_to_or_fail<'a>(&'a self, to: &Idx) -> &'a Result;
    fn slice_or_fail<'a>(&'a self, from: &Idx, to: &Idx) -> &'a Result;
}

// and similar for SliceMut...
```

The index and slice traits work somewhat differently. For
`Index`/`IndexMut`, the return value is *implicitly* dereferenced, so
that notation like `v[i] = 3` makes sense. If you want to get your
hands on the actual reference, you usually need an explicit `&`, for
example `&v[i]` or `&mut v[i]` (the compiler determines whether to use
`Index` or `IndexMut` by context). This follows the C notational
tradition.

Slice notation, on the other hand, does *not* automatically dereference
and so requires a special `mut` marker: `v[mut 1..]`.

For both of these traits, the indexes themselves are taken by
reference, and the compiler automatically introduces a `&` (so you
write `v[3]` not `v[&3]`).

### Proposed design

This RFC proposes to refactor the slice design into more modular
components, which as a side-product will make slicing automatically
dereference the result (consistently with indexing). The latter is
desirable because `&mut v[1..]` is more consistent with the rest of
the language than `v[mut 1..]` (and also makes the borrowing semantics
more explicit).

#### Index revisions

In the new design, the index traits take the index by value and the
compiler no longer introduces a silent `&`. This follows the same
design as for e.g. `Add` above, and for much the same reasons. That
means in particular that it will be possible to write `map["key"]`
rather than `map[*"key"]` when using a map with `String` keys, and
will still be possible to write `v[3]` for vectors. In addition, the
`Result` becomes an associated type, again following the same design
outlined above:

```rust
pub trait Index<Idx> for Sized?  {
    type Sized? Result;
    fn index<'a>(&'a self, index: Idx) -> &'a Result;
}

pub trait IndexMut<Idx> for Sized? {
    type Sized? Result;
    fn index_mut<'a>(&'a mut self, index: Idx) -> &'a mut Result;
}
```

In addition, this RFC proposes another trait, `IndexSet`, that is used for `expr[i] = expr`:

```rust
pub trait IndexSet<Idx> {
    type Val;
    fn index_set<'a>(&'a mut self, index: Idx, val: Val);
}
```

(This idea is borrowed from
[@sfackler's earlier RFC](https://github.com/rust-lang/rfcs/pull/159/files).)

The motivation for this trait is cases like `map["key"] = val`, which
should correspond to an *insertion* rather than a mutable lookup. With
today's setup, that expression would result in a panic if "key" was
not already present in the map.

Of course, `IndexSet` and `IndexMut` overlap, since `expr[i] = expr`
could be interpreted using either. Some types may implement `IndexSet`
but not `IndexMut` (for example, if it doesn't make sense to produce
an interior reference). But for types providing both, the compiler
will use `IndexSet` to interpret the `expr[i] = expr` syntax. (You can
always get `IndexMut` by instead writing `* &mut expr[i] = expr`, but
this will likely be extremely rare.)

#### Slice revisions

The changes to slice notation are more radical: this RFC proposes to
remove the slice traits altogether! The replacement is to introduce
*range notation* and overload indexing on it.

The current slice notation allows you to write `v[i..j]`, `v[i..]`,
`v[..j]` and `v[]`. The idea for handling the first three is to add
the following desugaring:

```rust
i..j  ==>  Range(i, j)
i..   ==>  RangeFrom(i)
..j   ==>  RangeTo(j)

where

struct Range<Idx>(Idx, Idx);
struct RangeFrom<Idx>(Idx);
struct RangeTo<Idx>(Idx);
```

Then, to implement slice notation, you just implement `Index`/`IndexMut` with
`Range`, `RangeFrom`, and `RangeTo` index types.

This cuts down on the number of special traits and machinery. It makes
indexing and slicing more consistent (since both will implicitly deref
their result); you'll write `&mut v[1..]` to get a mutable slice. It
also opens the door to other uses of the range notation:

```
for x in 1..100 { ... }
```

because the refactored design is more modular.

What about `v[]` notation? The proposal is to desugar this to
`v[FullRange]` where `struct FullRange;`.

Note that `..` is already used in a few places in the grammar, notably
fixed length arrays and functional record update. The former is at the
type level, however, and the latter is not ambiguous: `Foo { a: x,
.. bar}` since the `.. bar` component will never be parsed as an
expression.

## Special traits

Finally, there are a few "special" traits that hook into the compiler
in various ways that go beyond basic operator overlaoding.

### `Deref` and `DerefMut`

The `Deref` and `DerefMut` traits are used for overloading
dereferencing, typically for smart pointers.

The current traits look like so:

```rust
pub trait Deref<Sized? Result> {
    fn deref<'a>(&'a self) -> &'a Result;
}
```

but the `Result` type should become an associated type, dictating that
a smart pointer can only deref to a single other type (which is also
needed for inference and other magic around deref):

```rust
pub trait Deref {
    type Sized? Result;
    fn deref<'a>(&'a self) -> &'a Result;
}
```

### `Drop`

This RFC proposes no changes to the `Drop` trait.

### Closure traits

This RFC proposes no changes to the closure traits. The current design looks like:

```rust
pub trait Fn<Args, Result> {
    fn call(&self, args: Args) -> Result;
}
```

and, given the way that multidispatch has worked out, it is safe and
more flexible to keep both `Args` and `Result` as input types (which
means that custom implementations could overload on either). In
particular, the sugar for these traits requires writing all of these
types anyway.

These traits should *not* be exposed as `#[stable]` for 1.0, meaning
that you will not be able to implement or use them directly from the
[stable release channel](http://blog.rust-lang.org/2014/10/30/Stability.html). There
are a few reasons for this. For one, when bounding by these traits you
generally want to use the sugar `Fn (T, U) -> V` instead, which will
be stable. Keeping the traits themselves unstable leaves us room to
change their definition to support
[variadic generics](https://github.com/rust-lang/rfcs/issues/376) in
the future.

# Drawbacks

The main drawback is that implementing the above will take a bit of
time, which is something we're currently very short on. However,
stabilizing `cmp` and `ops` has always been part of the plan, and has
to be done for 1.0.

# Alternatives

## Comparison traits

We could pursue a more aggressive change to the comparison traits by
not having `PartialOrd` be a super trait of `Ord`, but instead
providing a blanket `impl` for `PartialOrd` for any `T:
Ord`. Unfortunately, this design poses some problems when it comes to
things like tuples, which want to provide `PartialOrd` and `Ord` if
all their components do: you would end up with overlapping
`PartialOrd` `impl`s. It's possible to work around this, but at the
expense of additional language features (like "negative bounds", the
ability to make an `impl` apply only when certain things are *not*
true).

Since it's unlikely that these other changes can happen in time for
1.0, this RFC takes a more conservative approach.

## Slicing

We may want to drop the `[]` notation. This notation was introduced to
improve ergonomics (from `foo(v.as_slice())` to `foo(v[]`), but now
that [collections reform](https://github.com/rust-lang/rfcs/pull/235)
is starting to land we can instead write `foo(&*v)`. If we also had
[deref coercions](https://github.com/rust-lang/rfcs/pull/241), that
would be just `foo(&v)`.

While `&*v` notation is more ergonomic than `v.as_slice()`, it is also
somewhat intimidating notation for a situation that newcomers to the
language are likely to face quickly.

In the opinion of this RFC author, we should either keep `[]`
notation, or provide deref coercions so that you can just say `&v`.

# Unresolved questions

In the long run, we should support overloading of operators like `+=`
which often have a more efficient implementation than desugaring into
a `+` and an `=`. However, this can be added backwards-compatibly and
is not significantly blocking library stabilization, so this RFC
postpones consideration until a later date.
