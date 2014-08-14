- Start Date: (fill me in with today's date, 2014-08-12)
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

This RFC adds *overloaded slice notation*:

- `foo[]` for `foo.as_slice()`
- `foo[n..m]` for `foo.slice(n, m)`
- `foo[n..]` for `foo.slice_from(n)`
- `foo[..m]` for `foo.slice_to(m)`
- `mut` variants of all the above

via two new traits, `Slice` and `SliceMut`.

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

## Rationale for the notation

The choice of square brackets for slicing is straightforward: it matches our
indexing notation, and slicing and indexing are closely related.

Some other languages (like Python and Go -- and Fortran) use `:` rather than
`..` in slice notation. The choice of `..` here is influenced by its use
elsewhere in Rust, for example for fixed-length array types `[T, ..n]`. The `..`
for slicing has precedent in Perl and D.

See [Wikipedia](http://en.wikipedia.org/wiki/Array_slicing) for more on the
history of slice notation in programming languages.

# Drawbacks

The main drawback is the increase in complexity of the language syntax. This
seems minor, especially since the notation here is essentially "finishing" what
was started with the `Index` trait.

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
