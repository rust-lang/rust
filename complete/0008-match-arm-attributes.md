- Start Date: 2014-03-20
- RFC PR: [rust-lang/rfcs#49](https://github.com/rust-lang/rfcs/pull/49)
- Rust Issue: [rust-lang/rust#12812](https://github.com/rust-lang/rust/issues/12812)

# Summary

Allow attributes on match arms.

# Motivation

One sometimes wishes to annotate the arms of match statements with
attributes, for example with conditional complilation `#[cfg]`s or
with branch weights (the latter is the most important use).

For the conditional compilation, the work-around is duplicating the
whole containing function with a `#[cfg]`. A case study is
[sfackler's bindings to OpenSSL](https://github.com/sfackler/rust-openssl),
where many distributions remove SSLv2 support, and so that portion of
Rust bindings needs to be conditionally disabled. The obvious way to
support the various different SSL versions is an enum

```rust
pub enum SslMethod {
    #[cfg(sslv2)]
    /// Only support the SSLv2 protocol
    Sslv2,
    /// Only support the SSLv3 protocol
    Sslv3,
    /// Only support the TLSv1 protocol
    Tlsv1,
    /// Support the SSLv2, SSLv3 and TLSv1 protocols
    Sslv23,
}
```

However, all `match`s can only mention `Sslv2` when the `cfg` is
active, i.e. the following is invalid:

```rust
fn name(method: SslMethod) -> &'static str {
    match method {
        Sslv2 => "SSLv2",
        Sslv3 => "SSLv3",
        _ => "..."
    }
}
```

A valid method would be to have two definitions: `#[cfg(sslv2)] fn
name(...)` and `#[cfg(not(sslv2)] fn name(...)`. The former has the
`Sslv2` arm, the latter does not. Clearly, this explodes exponentially
for each additional `cfg`'d variant in an enum.

Branch weights would allow the careful micro-optimiser to inform the
compiler that, for example, a certain match arm is rarely taken:

```rust
match foo {
    Common => {}
    #[cold]
    Rare => {}
}
```


# Detailed design

Normal attribute syntax, applied to a whole match arm.

```rust
match x {
    #[attr]
    Thing => {}

    #[attr]
    Foo | Bar => {}

    #[attr]
    _ => {}
}
```

# Alternatives

There aren't really any general alternatives; one could probably hack
around matching on conditional enum variants with some macros and
helper functions to share as much code as possible; but in general
this won't work.

# Unresolved questions

Nothing particularly.
