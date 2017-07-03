- Feature Name: as_cell
- Start Date: 2016-11-13
- RFC PR: https://github.com/rust-lang/rfcs/pull/1789
- Rust Issue: https://github.com/rust-lang/rust/issues/43038

# Summary
[summary]: #summary

- Change `Cell<T>` to allow `T: ?Sized`.
- Guarantee that `T` and `Cell<T>` have the same memory layout.
- Enable the following conversions through the std lib:
  - `&mut T -> &Cell<T> where T: ?Sized`
  - `&Cell<[T]> -> &[Cell<T>]`

> Note: https://github.com/rust-lang/rfcs/pull/1651 has been accepted recently,
> so no `T: Copy` bound is needed anymore.

# Motivation
[motivation]: #motivation

Rust's iterators offer a safe, fast way to iterate over collections while avoiding
additional bound checks.

However, due to the borrow checker, they run into issues if we try to have
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

...but this reintroduces potential bound-checking costs.

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

- The need to change the definition of the data structure containing the data
  (Which is not always possible because it might come from external code).
- Loss of the ability to directly hand out `&T` and `&mut T` references to the data.

This RFC proposes a way to address these in cases where `Cell<T>`
could be used by introducing simple conversions functions
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

The core of this proposal is the ability to convert a `&T` to a `&Cell<T>`,
so in order for it to be safe, __it needs to be guaranteed that
`T` and `Cell<T>` have the same memory layout__, and that there are no codegen
issues based on viewing a reference to a type that does not contain a
`UnsafeCell` as a reference to a type that does contain a `UnsafeCell`.

As far as the author is aware, both should already implicitly
fall out of the semantic of `Cell` and Rusts/llvms notion of aliasing:

- `Cell` is safe interior mutability based on memcopying the `T`,
  and thus does not need additional fields or padding.
- `&mut T -> &U` is a sub borrow, which prevents access to the original `&mut T`
  for its duration, thus no aliasing.

## Std library

### `from_mut`

We add a constructor to the cell API that enables the `&mut T -> &Cell<T>`
conversion, implemented with the equivalent of a `transmute()` of the two
pointers:

```rust
impl<T> Cell<T> {
    fn from_mut<'a>(t: &'a mut T) -> &'a Cell<T> {
        unsafe {
            &*(t as *mut T as *const Cell<T>)
        }
    }
}
```

In the future this could also be provided through `AsRef`, `Into` or `From`
impls.

### Unsized `Cell<T>`

We extend `Cell<T>` to allow `T: ?Sized`, and move all compatible methods
to a less restricted impl block:

```rust
pub struct Cell<T: ?Sized> {
    value: UnsafeCell<T>,
}

impl<T: ?Sized> Cell<T> {
    pub fn as_ptr(&self) -> *mut T;
    pub fn get_mut(&mut self) -> &mut T;
    pub fn from_mut(value: &mut T) -> &Cell<T>;
}
```

This is purely done to enable cell slicing below, and should otherwise have no
effect on any existing code.

### Cell Slicing

We enable a conversion from `&Cell<[T]>` to `&[Cell<T>]`. This seems like it violates
the "no interior references" API of `Cell` at first glance, but is actually
safe:

- A slice represents a number of elements next to each other.
  Thus, if `&mut T -> &Cell<T>` is ok, then `&mut [T] -> &[Cell<T>]` would be as well.
  `&mut [T] -> &Cell<[T]>` follows from `&mut T -> &Cell<T>` through substitution,
  so `&Cell<[T]> <-> &[Cell<T>]` has to be valid.
- The API of a `Cell<T>` is to allow internal mutability through single-threaded
  memcopies only. Since a memcopy is just a copy of all bits that make up a type,
  it does not matter if we logically do a memcopy to all elements of a slice
  through a `&Cell<[T]>`, or just a memcopy to a single element through a
  `&Cell<T>`.
- Yet another way to look at it is that if we created a `&mut T` to each element
  of a `&mut [T]`, and converted each of them to a `&Cell<T>`, their addresses
  would allow "stitching" them back together to a single `&[Cell<T>]`

For convenience, we expose this conversion by implementing `Index` for `Cell<[T]>`:

```rust
impl<T> Index<RangeFull> for Cell<[T]> {
    type Output = [Cell<T>];

    fn index(&self, _: RangeFull) -> &[Cell<T>] {
        unsafe {
            &*(self as *const Cell<[T]> as *const [Cell<T>])
        }
    }
}

impl<T> Index<Range<usize>> for Cell<[T]> {
    type Output = [Cell<T>];

    fn index(&self, idx: Range<usize>) -> &[Cell<T>] {
        &self[..][idx]
    }
}

impl<T> Index<RangeFrom<usize>> for Cell<[T]> {
    type Output = [Cell<T>];

    fn index(&self, idx: RangeFrom<usize>) -> &[Cell<T>] {
        &self[..][idx]
    }
}

impl<T> Index<RangeTo<usize>> for Cell<[T]> {
    type Output = [Cell<T>];

    fn index(&self, idx: RangeTo<usize>) -> &[Cell<T>] {
        &self[..][idx]
    }
}

impl<T> Index<usize> for Cell<[T]> {
    type Output = Cell<T>;

    fn index(&self, idx: usize) -> &Cell<T> {
        &self[..][idx]
    }
}
```

Using this, the motivation example can be written as such:

```rust
let mut v: Vec<i32> = ...;

// convert the mutable borrow
let v_slice: &[Cell<T>] = &Cell::from_mut(&mut v[..])[..];

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

## Possible extensions

The proposal only covers the base case `&mut T -> &Cell<T>`
and the trivially implementable extension to `[T]`,
but in theory this conversion could be enabled for
many "higher level mutable reference" types, like for example
mutable iterators (with the goal of making them cloneable through this).

See https://play.rust-lang.org/?gist=d012cebf462841887323185cff8ccbcc&version=stable&backtrace=0 for
an example implementation and a more complex use case, and
https://crates.io/crates/alias for an existing crate providing these features.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

> What names and terminology work best for these concepts and why?
How is this idea best presented—as a continuation of existing Rust patterns, or as a wholly new one?

The API could be described as "temporarily opting-in to internal mutability".
It would be a more flexible continuation of the existing usage of `Cell<T>`
since the `Cell<T>` no longer needs to exist in the original location if
you have mutable access to it.

> Would the acceptance of this proposal change how Rust is taught to new users at any level?
How should this feature be introduced and taught to existing Rust users?

As it is, the API just provides a few neat conversion functions. Nevertheless,
with the legalization of the `&mut T -> &Cell<T>` conversion there is the
potential for a major change in how accessors to data structures are provided:

In todays Rust, there are generally three different ways:
- Owned access that starts off with a `T` and yield `U`.
- Shared borrowed access that starts off with a `&T` and yields `&U`.
- Mutable borrowed access that starts off with a `&mut T` and yields `&mut U`.

With this change, it would be possible in many cases to add a fourth accessor:

- Shared borrowed cell access that starts off with a `&mut T` and yields `&Cell<U>`.

For example, today there exist:

- `Vec<T> -> std::vec::IntoIter<T>`, which yields `T` values and is cloneable.
- `&[T] -> std::slice::Iter<T>`, which yields `&T` values and is
  cloneable because it does a shared borrow.
- `&mut [T] -> std::slice::IterMut<T>`, which yields `&mut T` values and is
  not cloneable because it does a mutable borrow.

We could then add a fourth iterator like this:

- `&mut [T] -> std::slice::CellIter<T>`, which yields `&Cell<T>` values and is
  cloneable because it does a shared borrow.

So there is the potential that we go away from teaching the "rule of three"
of ownership and change it to a "rule of four".

> What additions or changes to the Rust Reference, _The Rust Programming Language_, and/or _Rust by Example_ does it entail?

- The reference should explain that the `&mut T -> &Cell<T>` conversion,
  or specifically the `&mut T -> &UnsafeCell<T>` conversion is fine.
- The book could use the API introduced here if it talks about internal mutability,
  and use it as a "temporary opt-in" example.
- Rust by Example could have a few basic examples of situations where this API
  is useful, eg the ones mention in the motivation section above.

# Drawbacks
[drawbacks]: #drawbacks

> Why should we *not* do this?

- More complexity around the `Cell` API.
- `T` -> `Cell<T>` transmute compatibility might not be a desired guarantee.

# Alternatives
[alternatives]: #alternatives

## Removing cell slicing

Instead of allowing unsized types in `Cell` and adding the `Index` impls,
there could just be a single `&mut [T] -> &[Cell<T>]` conversions function:

```rust
impl<T> Cell<T> {
    /// [...]

    fn from_mut_slice<'a>(t: &'a mut [T]) -> &'a [Cell<T>] {
        unsafe {
            &*(t as *mut [T] as *const [Cell<T>])
        }
    }
}
```

Usage:

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

This would be less modular than the `&mut [T] -> &Cell<[T]> -> &[Cell<T>]`
conversions steps, while still offering essentially the same API.

## Just the language guarantee

The conversion could be guaranteed as correct, but not be provided by std
itself. This would serve as legitimization of external implementations like
[alias](https://crates.io/crates/alias).

## No guarantees

If the safety guarantees of the conversion can not be granted,
code would have to use direct indexing as today, with either possible
bound checking costs or the use of unsafe code to avoid them.

## Replacing `Index` impls with `Deref`

Instead of the `Index` impls, have only this `Deref` impl:

```rust
impl<T> Deref for Cell<[T]> {
    type Target = [Cell<T>];

    fn deref(&self) -> &[Cell<T>] {
        unsafe {
            &*(self as *const Cell<[T]> as *const [Cell<T>])
        }
    }
}
```

Pro:

- Automatic conversion due to deref coercions and auto deref.
- Less redundancy since we don't repeat the slicing impls of `[T]`.

Cons:

- `Cell<[T]> -> [Cell<T>]` conversion does not seem like a good usecase
  for `Deref`, since `Cell<[T]>` isn't a smartpointer.

## Cast to `&mut Cell<T>` instead of `&Cell<T>`

Nothing that makes the `&mut T -> &Cell<T>` conversion safe would prevent
`&mut T -> &mut Cell<T>` from being safe either, and the latter can be
trivially turned into a `&Cell<T>` while also allowing mutable access - eg to
call `Cell::as_mut()` to conversion back again.

Similar to that, there could also be a way to turn a `&mut [Cell<T>]` back
into a `&mut [T]`.

However, this does not seem to be actually useful since the only reason to use
this API is to make use of shared internal mutability.

## Exposing the functions differently

Instead of `Cell` constructors, we could just have freestanding functions
in, say, `std::cell`:

```rust
fn ref_as_cell<T>(t: &mut T) -> &Cell<T> {
    unsafe {
        &*(t as *mut T as *const Cell<T>)
    }
}

fn cell_slice<T>(t: &Cell<[T]>) -> &[Cell<T>] {
    unsafe {
        &*(t as *const Cell<[T]> as *const [Cell<T>])
    }
}
```

On the opposite spectrum, and should this feature end up being used
somewhat commonly,
we could provide the conversions by dedicated traits,
possibly in the prelude, or use the std coherence hack to implement
them directly on `&mut T` and `&mut [T]`:

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
```

But given the issues of adding methods to pointer-like types,
this approach in general would probably be not a good idea
(See the situation with `Rc` and `Arc`).

# Unresolved questions
[unresolved]: #unresolved-questions

None so far.
