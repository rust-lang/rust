- Feature Name: `result_expect`
- Start Date: 2015-05-13
- RFC PR: [rust-lang/rfcs#1119](https://github.com/rust-lang/rfcs/pull/1119)
- Rust Issue: [rust-lang/rust#25359](https://github.com/rust-lang/rust/pull/25359)

# Summary

Add an `expect` method to the Result type, bounded to `E: Debug`

# Motivation

While `Result::unwrap` exists, it does not allow annotating the panic message with the operation
attempted (e.g. what file was being opened). This is at odds to 'Option' which includes both
`unwrap` and `expect` (with the latter taking an arbitrary failure message).

# Detailed design

Add a new method to the same `impl` block as `Result::unwrap` that takes a `&str` message and
returns `T` if the `Result` was `Ok`. If the `Result` was `Err`, it panics with both the provided
message and the error value.

The format of the error message is left undefined in the documentation, but will most likely be
the following

```
panic!("{}: {:?}", msg, e)
```

# Drawbacks

- It involves adding a new method to a core rust type.
- The panic message format is less obvious than it is with `Option::expect` (where the panic message is the message passed)

# Alternatives

- We are perfectly free to not do this.
- A macro could be introduced to fill the same role (which would allow arbitrary formatting of the panic message).

# Unresolved questions

Are there any issues with the proposed format of the panic string?
