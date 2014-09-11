- Start Date: 2014-09-11
- RFC PR #: [rust-lang/rfcs#198](https://github.com/rust-lang/rfcs/pull/198)
- Rust Issue #: [rust-lang/rust#17177](https://github.com/rust-lang/rust/issues/17177)

# Summary

This RFC adds *overloaded slice notation*:

- `foo[]` for `foo.as_slice()`
- `foo[n..m]` for `foo.slice(n, m)`
- `foo[n..]` for `foo.slice_from(n)`
- `foo[..m]` for `foo.slice_to(m)`
- `mut` variants of all the above

via two new traits, `Slice` and `SliceMut`.

It also changes the notation for range `match` patterns to `...`, to
signify that they are inclusive whereas `..` in slices are exclusive.

# Motivation

There are two primary motivations for introducing this feature.

### Ergonomics

Slicing operations, especially `as_slice`, are a very common and basic thing to
do with vectors, and potentially many other kinds of containers.  We already
have notation for indexing via the `Index` trait, and this RFC is essentially a
continuation of that effort.

The `as_slice` operator is particularly important. Since we've moved away from
auto-slicing in coercions, explicit `as_slice` calls have become extremely
common, and are one of the
[leading ergonomic/first impression](https://github.com/rust-lang/rust/issues/14983)
problems with the language. There are a few other approaches to address this
particular problem, but these alternatives have downsides that are discussed
below (see "Alternatives").

### Error handling conventions

We are gradually moving toward a Python-like world where notation like `foo[n]`
calls `fail!` when `n` is out of bounds, while corresponding methods like `get`
return `Option` values rather than failing. By providing similar notation for
slicing, we open the door to following the same convention throughout
vector-like APIs.

# Detailed design

The design is a straightforward continuation of the `Index` trait design. We
introduce two new traits, for immutable and mutable slicing:

```rust
trait Slice<Idx, S> {
    fn as_slice<'a>(&'a self) -> &'a S;
    fn slice_from(&'a self, from: Idx) -> &'a S;
    fn slice_to(&'a self, to: Idx) -> &'a S;
    fn slice(&'a self, from: Idx, to: Idx) -> &'a S;
}

trait SliceMut<Idx, S> {
    fn as_mut_slice<'a>(&'a mut self) -> &'a mut S;
    fn slice_from_mut(&'a mut self, from: Idx) -> &'a mut S;
    fn slice_to_mut(&'a mut self, to: Idx) -> &'a mut S;
    fn slice_mut(&'a mut self, from: Idx, to: Idx) -> &'a mut S;
}
```

(Note, the mutable names here are part of likely changes to naming conventions
that will be described in a separate RFC).

These traits will be used when interpreting the following notation:

*Immutable slicing*

- `foo[]` for `foo.as_slice()`
- `foo[n..m]` for `foo.slice(n, m)`
- `foo[n..]` for `foo.slice_from(n)`
- `foo[..m]` for `foo.slice_to(m)`

*Mutable slicing*

- `foo[mut]` for `foo.as_mut_slice()`
- `foo[mut n..m]` for `foo.slice_mut(n, m)`
- `foo[mut n..]` for `foo.slice_from_mut(n)`
- `foo[mut ..m]` for `foo.slice_to_mut(m)`

Like `Index`, uses of this notation will auto-deref just as if they were method
invocations. So if `T` implements `Slice<uint, [U]>`, and `s: Smaht<T>`, then
`s[]` compiles and has type `&[U]`.

Note that slicing is "exclusive" (so `[n..m]` is the interval `n <= x
< m`), while `..` in `match` patterns is "inclusive". To avoid
confusion, we propose to change the `match` notation to `...` to
reflect the distinction. The reason to change the notation, rather
than the interpretation, is that the exclusive (respectively
inclusive) interpretation is the right default for slicing
(respectively matching).

## Rationale for the notation

The choice of square brackets for slicing is straightforward: it matches our
indexing notation, and slicing and indexing are closely related.

Some other languages (like Python and Go -- and Fortran) use `:` rather than
`..` in slice notation. The choice of `..` here is influenced by its use
elsewhere in Rust, for example for fixed-length array types `[T, ..n]`. The `..`
for slicing has precedent in Perl and D.

See [Wikipedia](http://en.wikipedia.org/wiki/Array_slicing) for more on the
history of slice notation in programming languages.

### The `mut` qualifier

It may be surprising that `mut` is used as a qualifier in the proposed
slice notation, but not for the indexing notation. The reason is that
indexing includes an implicit dereference. If `v: Vec<Foo>` then
`v[n]` has type `Foo`, not `&Foo` or `&mut Foo`. So if you want to get
a mutable reference via indexing, you write `&mut v[n]`. More
generally, this allows us to do resolution/typechecking prior to
resolving the mutability.

This treatment of `Index` matches the C tradition, and allows us to
write things like `v[0] = foo` instead of `*v[0] = foo`.

On the other hand, this approach is problematic for slicing, since in
general it would yield an unsized type (under DST) -- and of course,
slicing is meant to give you a fat pointer indicating the size of the
slice, which we don't want to immediately deref. But the consequence
is that we need to know the mutability of the slice up front, when we
take it, since it determines the type of the expression.

# Drawbacks

The main drawback is the increase in complexity of the language syntax. This
seems minor, especially since the notation here is essentially "finishing" what
was started with the `Index` trait.

## Limitations in the design

Like the `Index` trait, this forces the result to be a reference via
`&`, which may rule out some generalizations of slicing.

One way of solving this problem is for the slice methods to take
`self` (by value) rather than `&self`, and in turn to implement the
trait on `&T` rather than `T`. Whether this approach is viable in the
long run will depend on the final rules for method resolution and
auto-ref.

In general, the trait system works best when traits can be applied to
types `T` rather than borrowed types `&T`. Ultimately, if Rust gains
higher-kinded types (HKT), we could change the slice type `S` in the
trait to be higher-kinded, so that it is a *family* of types indexed
by lifetime. Then we could replace the `&'a S` in the return value
with `S<'a>`. It should be possible to transition from the current
`Index` and `Slice` trait designs to an HKT version in the future
without breaking backwards compatibility by using blanket
implementations of the new traits (say, `IndexHKT`) for types that
implement the old ones.

# Alternatives

For improving the ergonomics of `as_slice`, there are two main alternatives.

## Coercions: auto-slicing

One possibility would be re-introducing some kind of coercion that automatically
slices.
We used to have a coercion from (in today's terms) `Vec<T>` to
`&[T]`. Since we no longer coerce owned to borrowed values, we'd probably want a
coercion `&Vec<T>` to `&[T]` now:

```rust
fn use_slice(t: &[u8]) { ... }

let v = vec!(0u8, 1, 2);
use_slice(&v)           // automatically coerce here
use_slice(v.as_slice()) // equivalent
```

Unfortunately, adding such a coercion requires choosing between the following:

* Tie the coercion to `Vec` and `String`. This would reintroduce special
  treatment of these otherwise purely library types, and would mean that other
  library types that support slicing would not benefit (defeating some of the
  purpose of DST).

* Make the coercion extensible, via a trait. This is opening pandora's box,
  however: the mechanism could likely be (ab)used to run arbitrary code during
  coercion, so that any invocation `foo(a, b, c)` might involve running code to
  pre-process each of the arguments. While we may eventually want such
  user-extensible coercions, it is a *big* step to take with a lot of potential
  downside when reasoning about code, so we should pursue more conservative
  solutions first.

## Deref

Another possibility would be to make `String` implement `Deref<str>` and
`Vec<T>` implement `Deref<[T]>`, once DST lands. Doing so would allow explicit
coercions like:

```rust
fn use_slice(t: &[u8]) { ... }

let v = vec!(0u8, 1, 2);
use_slice(&*v)          // take advantage of deref
use_slice(v.as_slice()) // equivalent
```

There are at least two downsides to doing so, however:

* It is not clear how the method resolution rules will ultimately interact with
  `Deref`. In particular, a leading proposal is that for a smart pointer `s: Smaht<T>`
  when you invoke `s.m(...)` only *inherent* methods `m` are considered for
  `Smaht<T>`; *trait* methods are only considered for the maximally-derefed
  value `*s`.

  With such a resolution strategy, implementing `Deref` for `Vec` would make it
  impossible to use trait methods on the `Vec` type except through UFCS,
  severely limiting the ability of programmers to usefully implement new traits
  for `Vec`.

* The idea of `Vec` as a smart pointer around a slice, and the use of `&*v` as
  above, is somewhat counterintuitive, especially for such a basic type.

Ultimately, notation for slicing seems desireable on its own merits anyway, and
if it can eliminate the need to implement `Deref` for `Vec` and `String`, all
the better.
