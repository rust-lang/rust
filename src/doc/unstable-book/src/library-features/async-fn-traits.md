# `async_fn_traits`

See Also: [`fn_traits`](../library-features/fn-traits.md)

----

The `async_fn_traits` feature allows for implementation of the [`AsyncFn*`] traits
for creating custom closure-like types that return futures.

[`AsyncFn*`]: ../../std/ops/trait.AsyncFn.html

The main difference to the `Fn*` family of traits is that `AsyncFn` can return a future
that borrows from itself (`FnOnce::Output` has no lifetime parameters, while `AsyncFnMut::CallRefFuture` does).
