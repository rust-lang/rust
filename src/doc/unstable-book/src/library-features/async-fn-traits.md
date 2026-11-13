# `async_fn_traits`

The tracking issue for this feature is: [#146790]

[#146790]: https://github.com/rust-lang/rust/issues/146790

See also: [`fn_traits`](../library-features/fn-traits.md)

----

The [`AsyncFn*`] traits are stable for use as bounds on async callables.
The `async_fn_traits` feature enables their unstable methods and associated
types, allowing custom implementations and explicit bounds on the futures
they return.

Unlike `Fn` and `FnMut`, whose returned type cannot borrow from the callable,
`AsyncFn` and `AsyncFnMut` support futures that borrow from the callable.
This is represented by the lifetime parameter on `AsyncFnMut::CallRefFuture`.

[`AsyncFn*`]: ../../std/ops/trait.AsyncFn.html
