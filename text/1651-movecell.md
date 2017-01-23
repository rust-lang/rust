- Feature Name: move_cell
- Start Date: 2016-06-15
- RFC PR: https://github.com/rust-lang/rfcs/pull/1651
- Rust Issue: https://github.com/rust-lang/rust/issues/39264

# Summary
[summary]: #summary

Extend `Cell` to work with non-`Copy` types.

# Motivation
[motivation]: #motivation

It allows safe inner-mutability of non-`Copy` types without the overhead of `RefCell`'s reference counting.

The key idea of `Cell` is to provide a primitive building block to safely support inner mutability. This must be done while maintaining Rust's aliasing requirements for mutable references. Unlike `RefCell` which enforces this at runtime through reference counting, `Cell` does this statically by disallowing any reference (mutable or immutable) to the data contained in the cell.

While the current implementation only supports `Copy` types, this restriction isn't actually necessary to maintain Rust's aliasing invariants. The only affected API is the `get` function which, by design, is only usable with `Copy` types.

# Detailed design
[design]: #detailed-design

```rust
impl<T> Cell<T> {
    fn set(&self, val: T);
    fn replace(&self, val: T) -> T;
    fn into_inner(self) -> T;
}

impl<T: Copy> Cell<T> {
    fn get(&self) -> T;
}

impl<T: Default> Cell<T> {
    fn take(&self) -> T;
}
```

The `get` method is kept but is only available for `T: Copy`.

The `set` method is available for all `T`. It will need to be implemented by calling `replace` and dropping the returned value. Dropping the old value in-place is unsound since the `Drop` impl will hold a mutable reference to the cell contents.

The `into_inner` and `replace` methods are added, which allow the value in a cell to be read even if `T` is not `Copy`. The `get` method can't be used since the cell must always contain a valid value.

Finally, a `take` method is added which is equivalent to `self.replace(Default::default())`.

# Drawbacks
[drawbacks]: #drawbacks

It makes the `Cell` type more complicated.

`Cell` will only be able to derive traits like `Eq` and `Ord` for types that are `Copy`, since there is no way to non-destructively read the contents of a non-`Copy` `Cell`.

# Alternatives
[alternatives]: #alternatives

The alternative is to use the `MoveCell` type from crates.io which provides the same functionality.

# Unresolved questions
[unresolved]: #unresolved-questions

None
