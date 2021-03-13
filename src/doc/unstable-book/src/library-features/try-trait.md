# `try_trait`

The tracking issue for this feature is: [#42327]

[#42327]: https://github.com/rust-lang/rust/issues/42327

------------------------

This introduces a new trait `Try` for extending the `?` operator to types
other than `Result` (a part of [RFC 1859]).  The trait provides the canonical
way to _view_ a type in terms of a success/failure dichotomy.  This will
allow `?` to supplant the `try_opt!` macro on `Option` and the `try_ready!`
macro on `Poll`, among other things.

[RFC 1859]: https://github.com/rust-lang/rfcs/pull/1859

Here's an example implementation of the trait:

```rust,ignore (cannot-reimpl-Try)
/// A distinct type to represent the `None` value of an `Option`.
///
/// This enables using the `?` operator on `Option`; it's rarely useful alone.
#[derive(Debug)]
#[unstable(feature = "try_trait", issue = "42327")]
pub struct None { _priv: () }

#[unstable(feature = "try_trait", issue = "42327")]
impl<T> ops::Try for Option<T>  {
    type Ok = T;
    type Error = None;

    fn into_result(self) -> Result<T, None> {
        self.ok_or(None { _priv: () })
    }

    fn from_ok(v: T) -> Self {
        Some(v)
    }

    fn from_error(_: None) -> Self {
        None
    }
}
```

Note the `Error` associated type here is a new marker.  The `?` operator
allows interconversion between different `Try` implementers only when
the error type can be converted `Into` the error type of the enclosing
function (or catch block).  Having a distinct error type (as opposed to
just `()`, or similar) restricts this to where it's semantically meaningful.
