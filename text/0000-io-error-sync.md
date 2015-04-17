- Feature Name: `io_error_sync`
- Start Date: 2015-04-11
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add the `Sync` bound to `io::Error` by requiring that any wrapped custom errors
also conform to `Sync` in addition to `error::Error + Send`.

# Motivation

Adding the `Sync` bound to `io::Error` has 3 primary benefits:

* Values that contain `io::Error`s will be able to be `Sync`
* Perhaps more importantly, `io::Error` will be able to be stored in an `Arc`
* By using the above, a cloneable wrapper can be created that shares an
  `io::Error` using an `Arc` in order to simulate the old behavior of being able
  to clone an `io::Error`.

# Detailed design

The only thing keeping `io::Error` from being `Sync` today is the wrapped custom
error type `Box<error::Error+Send>`. Changing this to
`Box<error::Error+Send+Sync>` and adding the `Sync` bound to `io::Error::new()`
is sufficient to make `io::Error` be `Sync`. In addition, the relevant
`convert::From` impls that convert to `Box<error::Error+Send>` will be updated
to convert to `Box<error::Error+Send+Sync>` instead.

# Drawbacks

The only downside to this change is it means any types that conform to
`error::Error` and are `Send` but not `Sync` will no longer be able to be
wrapped in an `io::Error`. It's unclear if there's any types in the standard
library that will be impacted by this. Looking through the [list of
implementors][impls] for `error::Error`, here's all of the types that may be
affected:

* `io::IntoInnerError`: This type is only `Sync` if the underlying buffered
  writer instance is `Sync`. I can't be sure, but I don't believe we have any
  writers that are `Send` but not `Sync`. In addition, this type has a `From`
  impl that converts it to `io::Error` even if the writer is not `Send`.
* `sync::mpsc::SendError`: This type is only `Sync` if the wrapped value `T` is
  `Sync`. This is of course also true for `Send`. I'm not sure if anyone is
  relying on the ability to wrap a `SendError` in an `io::Error`.
* `sync::mpsc::TrySendError`: Same situation as `SendError`.
* `sync::PoisonError`: This type is already not compatible with `io::Error`
  because it wraps mutex guards (such as `sync::MutexGuard`) which are not
  `Send`.
* `sync::TryLockError`: Same situation as `PoisonError`.

So the only real question is about `sync::mpsc::SendError`. If anyone is relying
on the ability to convert that into an `io::Error` a `From` impl could be
added that returns an `io::Error` that is indistinguishable from a wrapped
`SendError`.

[impls]: http://doc.rust-lang.org/nightly/std/error/trait.Error.html

# Alternatives

Don't do this. Not adding the `Sync` bound to `io::Error` means `io::Error`s
cannot be stored in an `Arc` and types that contain an `io::Error` cannot be
`Sync`.

We should also consider whether we should go a step further and change
`io::Error` to use `Arc` instead of `Box` internally. This would let us restore
the `Clone` impl for `io::Error`.

# Unresolved questions

Should we add the `From` impl for `SendError`? There is no code in the rust
project that relies on `SendError` being converted to `io::Error`, and I'm
inclined to think it's unlikely for anyone to be relying on that, but I don't
know if there are any third-party crates that will be affected.
