- Start Date: 2014-03-20
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

Allow attributes on more places inside functions, such as match arms
and statements.

# Motivation

One sometimes wishes to annotate things inside functions with, for
example, lint `#[allow]`s, conditional compilation `#[cfg]`s, branch
weight hints and even extra semantic (or otherwise) annotations for
external tools.

For the lints, one can currently only activate lints at the level of
the function which is possibly larger than one needs, and so may allow
other "bad" things to sneak through accidentally. E.g.

```rust
#[allow(uppercase_variable)]
let L = List::new(); // lowercase looks like one or capital i
```

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

The sort of things one could do with other arbitrary annotations are

```rust
#[allowed_unsafe_actions(ffi)]
#[audited="2014-04-22"]
unsafe { ... }
```

and then have an external tool that checks that that `unsafe` block's
only unsafe actions are FFI, or a tool that lists blocks that have
been changed since the last audit or haven't been audited ever.


# Detailed design

Normal attribute syntax:

```rust
fn foo() {
    #[attr]
    let x = 1;

    #[attr]
    foo();

    #[attr]
    match x {
        #[attr]
        Thing => {}
    }

    #[attr]
    if foo {
    } else {
    }
}
```

# Alternatives

There aren't really any general alternatives; one could probably hack
around the conditional-enum-variants & matches with some macros and
helper functions to share as much code as possible; but in general
this won't work.

The other instances could be approximated with macros and helper
functions, but to an even lesser degree (e.g. how would one annotate a
general `unsafe` block).

# Unresolved questions

- Should one be able to annotate the `else` branch(es) of an `if`? e.g.

  ```rust
  if foo {
  } #[attr] else if bar {
  } #[attr] else {
  }
  ```

  or maybe

  ```rust
  if foo {
  } else #[attr] if bar {
  } else #[attr] {
  }
  ```
