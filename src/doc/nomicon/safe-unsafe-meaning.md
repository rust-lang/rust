% How Safe and Unsafe Interact

What's the relationship between Safe Rust and Unsafe Rust? How do they
interact?

The separation between Safe Rust and Unsafe Rust is controlled with the
`unsafe` keyword, which acts as an interface from one to the other. This is
why we can say Safe Rust is a safe language: all the unsafe parts are kept
exclusively behind the boundary.

The `unsafe` keyword has two uses: to declare the existence of contracts the
compiler can't check, and to declare that the adherence of some code to
those contracts has been checked by the programmer.

You can use `unsafe` to indicate the existence of unchecked contracts on
_functions_ and on _trait declarations_. On functions, `unsafe` means that
users of the function must check that function's documentation to ensure
they are using it in a way that maintains the contracts the function
requires. On trait declarations, `unsafe` means that implementors of the
trait must check the trait documentation to ensure their implementation
maintains the contracts the trait requires.

You can use `unsafe` on a block to declare that all constraints required
by an unsafe function within the block have been adhered to, and the code
can therefore be trusted. You can use `unsafe` on a trait implementation
to declare that the implementation of that trait has adhered to whatever
contracts the trait's documentation requires.

The standard library has a number of unsafe functions, including:

* `slice::get_unchecked`, which performs unchecked indexing, allowing
  memory safety to be freely violated.
* `mem::transmute` reinterprets some value as having a given type, bypassing
  type safety in arbitrary ways (see [conversions] for details).
* Every raw pointer to a sized type has an intrinstic `offset` method that
  invokes Undefined Behavior if the passed offset is not "in bounds" as
  defined by LLVM.
* All FFI functions are `unsafe` because the other language can do arbitrary
  operations that the Rust compiler can't check.

As of Rust 1.0 there are exactly two unsafe traits:

* `Send` is a marker trait (a trait with no API) that promises implementors are
  safe to send (move) to another thread.
* `Sync` is a marker trait that promises threads can safely share implementors
  through a shared reference.

Much of the Rust standard library also uses Unsafe Rust internally, although
these implementations are rigorously manually checked, and the Safe Rust
interfaces provided on top of these implementations can be assumed to be safe.

The need for all of this separation boils down a single fundamental property
of Safe Rust:

**No matter what, Safe Rust can't cause Undefined Behavior.**

The design of the safe/unsafe split means that Safe Rust inherently has to
trust that any Unsafe Rust it touches has been written correctly (meaning
the Unsafe Rust actually maintains whatever contracts it is supposed to
maintain). On the other hand, Unsafe Rust has to be very careful about
trusting Safe Rust.

As an example, Rust has the `PartialOrd` and `Ord` traits to differentiate
between types which can "just" be compared, and those that provide a total
ordering (where every value of the type is either equal to, greater than,
or less than any other value of the same type). The sorted map type
`BTreeMap` doesn't make sense for partially-ordered types, and so it
requires that any key type for it implements the `Ord` trait. However,
`BTreeMap` has Unsafe Rust code inside of its implementation, and this
Unsafe Rust code cannot assume that any `Ord` implementation it gets makes
sense. The unsafe portions of `BTreeMap`'s internals have to be careful to
maintain all necessary contracts, even if a key type's `Ord` implementation
does not implement a total ordering.

Unsafe Rust cannot automatically trust Safe Rust. When writing Unsafe Rust,
you must be careful to only rely on specific Safe Rust code, and not make
assumptions about potential future Safe Rust code providing the same
guarantees.

This is the problem that `unsafe` traits exist to resolve. The `BTreeMap`
type could theoretically require that keys implement a new trait called
`UnsafeOrd`, rather than `Ord`, that might look like this:

```rust
use std::cmp::Ordering;

unsafe trait UnsafeOrd {
    fn cmp(&self, other: &Self) -> Ordering;
}
```

Then, a type would use `unsafe` to implement `UnsafeOrd`, indicating that
they've ensured their implementation maintains whatever contracts the
trait expects. In this situation, the Unsafe Rust in the internals of
`BTreeMap` could trust that the key type's `UnsafeOrd` implementation is
correct. If it isn't, it's the fault of the unsafe trait implementation
code, which is consistent with Rust's safety guarantees.

The decision of whether to mark a trait `unsafe` is an API design choice.
Rust has traditionally avoided marking traits unsafe because it makes Unsafe
Rust pervasive, which is not desirable. `Send` and `Sync` are marked unsafe
because thread safety is a *fundamental property* that unsafe code can't
possibly hope to defend against in the way it could defend against a bad
`Ord` implementation. The decision of whether to mark your own traits `unsafe`
depends on the same sort of consideration. If `unsafe` code cannot reasonably
expect to defend against a bad implementation of the trait, then marking the
trait `unsafe` is a reasonable choice.

As an aside, while `Send` and `Sync` are `unsafe` traits, they are
automatically implemented for types when such derivations are provably safe
to do. `Send` is automatically derived for all types composed only of values
whose types also implement `Send`. `Sync` is automatically derived for all
types composed only of values whose types also implement `Sync`.

This is the dance of Safe Rust and Unsafe Rust. It is designed to make using
Safe Rust as ergonomic as possible, but requires extra effort and care when
writing Unsafe Rust. The rest of the book is largely a discussion of the sort
of care that must be taken, and what contracts it is expected of Unsafe Rust
to uphold.

[drop flags]: drop-flags.html
[conversions]: conversions.html

