- Feature Name: `try_borrow`
- Start Date: 2016-06-27
- RFC PR: [rust-lang/rfcs#1660](https://github.com/rust-lang/rfcs/pull/1660)
- Rust Issue: [rust-lang/rust#35070](https://github.com/rust-lang/rust/issues/35070)

# Summary
[summary]: #summary

Introduce non-panicking borrow methods on `RefCell<T>`.

# Motivation
[motivation]: #motivation

Whenever something is built from user input, for example a graph in which nodes
are `RefCell<T>` values, it is primordial to avoid panicking on bad input. The
only way to avoid panics on cyclic input in this case is a way to
conditionally-borrow the cell contents.

# Detailed design
[design]: #detailed-design

```rust
/// Returned when `RefCell::try_borrow` fails.
pub struct BorrowError { _inner: () }

/// Returned when `RefCell::try_borrow_mut` fails.
pub struct BorrowMutError { _inner: () }

impl RefCell<T> {
    /// Tries to immutably borrows the value. This returns `Err(_)` if the cell
    /// was already borrowed mutably.
    pub fn try_borrow(&self) -> Result<Ref<T>, BorrowError> { ... }

    /// Tries to mutably borrows the value. This returns `Err(_)` if the cell
    /// was already borrowed.
    pub fn try_borrow_mut(&self) -> Result<RefMut<T>, BorrowMutError> { ... }
}
```

# Drawbacks
[drawbacks]: #drawbacks

This departs from the fallible/infallible convention where we avoid providing
both panicking and non-panicking methods for the same operation.

# Alternatives
[alternatives]: #alternatives

The alternative is to provide a `borrow_state` method returning the state
of the borrow flag of the cell, i.e:

```rust
pub enum BorrowState {
    Reading,
    Writing,
    Unused,
}

impl<T> RefCell<T> {
    pub fn borrow_state(&self) -> BorrowState { ... }
}
```

See [the Rust tracking issue](https://github.com/rust-lang/rust/issues/27733)
for this feature.

# Unresolved questions
[unresolved]: #unresolved-questions

There are no unresolved questions.
