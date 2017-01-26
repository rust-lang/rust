- Feature Name: as_cell
- Start Date: 2016-11-13
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

For all `T`, define `Cell<T>` as having the same memory layout as `T` and
add conversion functions with the following signatures to std:

- `&mut T -> &Cell<T>`
- `&mut [T] -> &[Cell<T>]`

> Note: https://github.com/rust-lang/rfcs/pull/1651 has been accepted recently,
> so no `T: Copy` bound is needed anymore.

# Motivation
[motivation]: #motivation

Rust's iterators offer a safe, fast way to iterate over collections while avoiding
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

The alternative, short of changing the actual algorithms involved, is to use
internal mutability to enable safe mutations even with overlapping shared views into the data:

```rust
let v: Vec<Cell<i32>> = ...;

// example 1
for i in &v {
    for j in &v {
        j.set(f(i.get(), j.get()));
    }
}

// example 2
for (i, j) in v[n..].iter().zip(&v) {
    i.set(g(g.get()));
}

```

This has the advantages of allowing both bound-check free iteration and
aliasing references, but comes with restrictions that makes it not generally
applicable, namely:

- The need to modify the data structure containing the data (Which might not
  always be possible because it might come from external code).
- Loss of the ability to directly hand out `&T` and `&mut T` references to the data.
- Refcounting overhead in case of `RefCell<T>`.

This RFC proposes a way to address these in cases where `Cell<T>`
would be used by introducing simple conversions functions
to the standard library that allow the creation of shared borrowed
`Cell<T>`s from mutably borrowed `T`s.

This in turn allows the original data structure to remain unchanged,
while allowing to temporary opt-in to the `Cell` API as needed.
As an example, given `Cell::from_mut_slice(&mut [T]) -> &[Cell<T>]`,
the previous examples can be written as this:

```rust
let mut v: Vec<i32> = ...;

// convert the mutable borrow
let v_slice: &[Cell<T>] = Cell::from_mut_slice(&mut v);

// example 1
for i in v_slice {
    for j in v_slice {
        j.set(f(i.get(), j.get()));
    }
}

// example 2
for (i, j) in v_slice[n..].iter().zip(v_slice) {
    i.set(g(g.get()));
}

```

# Detailed design
[design]: #detailed-design

## Language

In order for this proposal to be safe, __it needs to be guaranteed that
`T` and `Cell<T>` have the same memory layout__, and that there are no codegen
issues based on viewing a reference to a `UnsafeCell`-less types as a
reference to a `UnsafeCell`-containing type.

As far as the author is aware, both should already implicitly
fall out of the semantic of `Cell` and llvms notion of aliasing:

- `Cell` is safe interior mutability based on memcopying the `T`,
  and thus does not need additional fields.
- `&mut T -> &U` is a sub borrow, which prevents access to the original `&mut T`
  for its duration, thus no aliasing.

However, we might have to add an explicit language guarantee about
the validity of this cast.

## Std library

Add conversions functions for `&mut T -> &Cell<T>` and `&mut [T] -> &[Cell<T>]`
to std, implemented with the equivalent of a simple `transmute()`.

As an initial design, `Cell` would provide them through additional constructors:

```rust
impl<T> Cell<T> {
    fn from_mut<'a>(t: &'a mut T) -> &'a Cell<T> {
        unsafe {
            &*(t as *mut T as *const Cell<T>)
        }
    }
    fn from_mut_slice<'a>(t: &'a mut [T]) -> &'a [Cell<T>] {
        unsafe {
            &*(t as *mut [T] as *const [Cell<T>])
        }
    }
}
```

It might also be possible to add `AsRef`, `Into` or `From`
versions of these conversions.

The proposal only covers the base case `&mut T -> &Cell<T>`
and the trivially implementable extension to `[T]`,
but in theory this conversion could be enabled for
many "higher level mutable reference" types, like for example
mutable iterators (with the goal of making them cloneable through this).

See https://play.rust-lang.org/?gist=d012cebf462841887323185cff8ccbcc&version=stable&backtrace=0 for
an example implementation and a more complex use case, and
https://crates.io/crates/alias for an existing crate providing these features.

# Drawbacks
[drawbacks]: #drawbacks

> Why should we *not* do this?

- More complexity around the `Cell` API.
- `T` -> `Cell<T>` transmute compatibility might not be a desired guarantee.

# Alternatives
[alternatives]: #alternatives

## Just the language guarantee

The cast could be guaranteed as correct, but not be provided by std
itself. This would serve as legitimization of external implementations like
[alias](https://crates.io/crates/alias).

## No guarantees

If the casting guarantees can not be granted,
code would have to use direct indexing as today, with either possible
bound checking costs or the use of unsafe code to avoid them.

## Exposing the functions differently

Instead of `Cell` constructors, we could just have freestanding functions
in, say, `std::cell`:

```rust
fn ref_as_cell<T>(t: &mut T) -> &Cell<T> {
    unsafe {
        &*(t as *mut T as *const Cell<T>)
    }
}

fn slice_as_cell<T>(t: &mut [T]) -> &[Cell<T>] {
    unsafe {
        &*(t as *mut [T] as *const [Cell<T>])
    }
}
```

On the opposite spectrum, and should this feature end up being used
somewhat commonly,
we could provide the conversions by dedicated traits,
possibly in the prelude, or use the std coherence hack to implement
them directly on `&mut T` and `& mut [T]`:

```rust
trait AsCell {
    type Cell;
    fn as_cell(self) -> Self::Cell;
}

impl<'a, T> AsCell for &'a mut T {
    type Cell = &'a Cell<T>;
    fn as_cell(self) -> Self::Cell {
        unsafe {
            &*(self as *mut T as *const Cell<T>)
        }
    }
}

impl<'a, T> AsCell for &'a mut [T] {
    type Cell = &'a [Cell<T>];
    fn as_cell(self) -> Self::Cell {
        unsafe {
            &*(self as *mut [T] as *const [Cell<T>])
        }
    }
}
```

But given the issues of adding methods to pointer-like types,
this approach in general would probably be not a good idea
(See the situation with `Rc` and `Arc`).

# Unresolved questions
[unresolved]: #unresolved-questions

Interactions with proposals like https://github.com/rust-lang/rfcs/pull/1651
are not investigated, but seem to only be beneficial by allowing the proposed
change to apply to more types.
