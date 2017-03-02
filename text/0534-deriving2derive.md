- Start Date: 2014-19-19
- RFC PR: [534](https://github.com/rust-lang/rfcs/pull/534)
- Rust Issue: [20362](https://github.com/rust-lang/rust/issues/20362)

# Summary

Rename the `#[deriving(Foo)]` syntax extension to `#[derive(Foo)]`.

# Motivation

Unlike our other verb-based attribute names, "deriving" stands alone as a
present participle. By convention our attributes prefer "warn" rather than
"warning", "inline" rather than "inlining", "test" rather than "testing", and so
on. We also have a trend against present participles in general, such as with
`Encoding` being changed to `Encode`.

It's also shorter to type, which is very important in a world without implicit
Copy implementations.

Finally, if I may be subjective, `derive(Thing1, Thing2)` simply reads better
than `deriving(Thing1, Thing2)`.

# Detailed design

Rename the `deriving` attribute to `derive`. This should be a very simple find-
and-replace.

# Drawbacks

Participles the world over will lament the loss of their only foothold in this
promising young language.
