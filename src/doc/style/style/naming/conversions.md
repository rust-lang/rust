% Conversions [Rust issue #7087]

> The guidelines below were approved by [rust issue #7087](https://github.com/rust-lang/rust/issues/7087).

> **[FIXME]** Should we provide standard traits for conversions? Doing
> so nicely will require
> [trait reform](https://github.com/rust-lang/rfcs/pull/48) to land.

Conversions should be provided as methods, with names prefixed as follows:

| Prefix | Cost | Consumes convertee |
| ------ | ---- | ------------------ |
| `as_` | Free | No |
| `to_` | Expensive | No |
| `into_` | Variable | Yes |

<p>
For example:

* `as_bytes()` gives a `&[u8]` view into a `&str`, which is a no-op.
* `to_owned()` copies a `&str` to a new `String`.
* `into_bytes()` consumes a `String` and yields the underlying
  `Vec<u8>`, which is a no-op.

Conversions prefixed `as_` and `into_` typically _decrease abstraction_, either
exposing a view into the underlying representation (`as`) or deconstructing data
into its underlying representation (`into`). Conversions prefixed `to_`, on the
other hand, typically stay at the same level of abstraction but do some work to
change one representation into another.

> **[FIXME]** The distinctions between conversion methods does not work
> so well for `from_` conversion constructors. Is that a problem?
