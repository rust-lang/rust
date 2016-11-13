- Feature Name: (fill me in with a unique ident, my_awesome_feature)
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

For `T: Copy`, add conversion functions with the following signatures to std:

- `&mut T -> &Cell<T>`
- `&mut [T] -> &[Cell<T>]`

# Motivation
[motivation]: #motivation

Rusts iterators offer a safe, fast way to iterate over collections while avoiding
additional bound checks.

However, due to the borrow checker, they run into issues if you try to have
more than one iterator into the same data structure while mutating elements in it.

Wanting to do this is not that unusual for many low level algorithms
that deal with integers, floats or similar primitive data types.

For example, an algorithm might...

- For each element, access each other element.
- For each element, access an element a number of elements before or after it.

Todays answer for algorithms like that is to fall back to C-style
for loops and indexing, which might look like this...

```rust

let v: Vec<i32> = ...;

// example 1
for i in 0..v.len() {
    for j in 0..v.len() {
        v[j] = f(v[i], v[j]);
    }
}

// example 2
for i in n..v.len() {
    v[i] = g(v[i - n]);
}

```

...but this reintroduces the bound-checking costs.

However, there exists a third alternative: Using internal mutability
to enable safe mutation even with only shared views into the data:

```rust
let v: Vec<Cell<i32>> = ...;

// example 1
for i in v.iter() {
    for j in v.iter() {
        j.set(f(i.get(), j.get()));
    }
}

// example 2
for (i, j) in v[n..].iter().zip(v.iter()) {
    i.set(g(g.get()));
}

```

This has the advantages of allowing both bound-check free iteration and
aliasing references, but comes with restrictions that makes it not generally
applicable, namely:

- The need to modify the data structure containing the data (Which might not
  always be possible because it might come from external code).
- Restriction to `Copy` types for `Cell`, and ref count overhead for `RefCell`.
- Loss of the ability to directly hand out `&T` and `&mut T` references to the data.

This RFC proposes a way to address the first and the last of the
previous restrictions by introducing simple conversions functions
to the standard library that allow the creation of `&Cell<T>`s from `&mut T`s.

This allows the original data structure to remain unchanged in type, which makes
this approach more applicable to more problems, and an answer to many of the
situations that require C-style for loops at the moment.

# Detailed design
[design]: #detailed-design

Add conversions functions for `&mut T -> &Cell<T>` and `&mut [T] -> &[Cell<T>]`
to std, implemented with the equivalent of a simple `transmute()`.

As an initial design, `std::cell` would provide them through this trait:

```rust
trait AsCell {
    type Cell;
    fn as_cell(self) -> Self::Cell;
}

impl<'a, T: Copy> AsCell for &'a mut T {
    type Cell = &'a Cell<T>;
    fn as_cell(self) -> Self::Cell {
        unsafe { mem::transmute(t) }
    }
}

impl<'a, T: Copy> AsCell for &'a mut [T] {
    type Cell = &'a [Cell<T>];
    fn as_cell(self) -> Self::Cell {
        unsafe { mem::transmute(t) }
    }
}
```

In theory it could get added to the prelude, but it should probably
follow `Cell`s lead there, which is currently not in there.

Method dispatch would work as expected with this design:

- `v.as_cell()` where `v: Vec` would correctly pick the `[T]` impl.
- `x.as_cell()` where `x` is a copyable type correctly picks the `T` impl.
- `a.as_cell()` where `a` is a copyable array would pick the `T` impl,
  and require explicit slicing with `a[..].as_cell()` to pick the `[T]` one.

However, if changes as proposed in https://github.com/rust-lang/rfcs/pull/1651 should
get implemented, the `Copy` bound might get relaxed or removed entirely,
which would affect the ergonomics here.

The proposed implementation only covers the base case `&mut T -> &Cell<T>`
and the trivially implementable extension to `[T]`,
but in theory this conversion could be enabled for
many "higher level mutable reference" types, like for example
mutable iterators (with the goal of making them cloneable through this).

In order for this proposal to be safe, it needs to be guaranteed that
`T` and `Cell<T>` have the same memory layout, and that there are no codegen
issues based on viewing a reference to a `UnsafeCell`-less types as a
reference to a `UnsafeCell`-containing type.

As far as the author is aware, both should already fall out of the semantic of
`Cell` and llvms notion of aliasing:

- `Cell` is safe interior mutability based on memcopying the `T`,
  and thus does not need additional fields.
- `&mut T -> &U` is a sub borrow, which prevents access to the original `&mut T`
  for its duration, thus no aliasing.

See https://play.rust-lang.org/?gist=d012cebf462841887323185cff8ccbcc&version=stable&backtrace=0 for
an example implementation and a more complex use case.

# Drawbacks
[drawbacks]: #drawbacks

> Why should we *not* do this?

- More complexity around the `Cell` API.
- `T` -> `Cell<T>` transmute compatibility might not be a desired guarantee.

# Alternatives
[alternatives]: #alternatives

Instead of a dedicated trait, the functionality could be provided
by standalone functions like this:

```rust
fn ref_as_cell<T: Copy>(t: &mut T) -> &Cell<T> {
    unsafe { mem::transmute(t) }
}

fn slice_as_cell<T: Copy>(t: &mut [T]) -> &[Cell<T>] {
    unsafe { mem::transmute(t) }
}
```

The cast could also be guaranteed as correct, but not be provided by std
itself, and rather be provided by an external crate.

Lastly, if the casting guarantees can not be granted,
code would have to use direct indexing as today, with either possible
bound checking costs or the use of unsafe code to avoid them.

# Unresolved questions
[unresolved]: #unresolved-questions

Interactions with proposals like https://github.com/rust-lang/rfcs/pull/1651
is unclear, but possibly beneficial.
