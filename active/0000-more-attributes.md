- Start Date: 2014-03-20
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

Allow attributes on more places inside functions, such as statements,
blocks and (possibly) expressions.

# Motivation

One sometimes wishes to annotate things inside functions with, for
example, lint `#[allow]`s, conditional compilation `#[cfg]`s, and even
extra semantic (or otherwise) annotations for external tools.

For the lints, one can currently only activate lints at the level of
the function which is possibly larger than one needs, and so may allow
other "bad" things to sneak through accidentally. E.g.

```rust
#[allow(uppercase_variable)]
let L = List::new(); // lowercase looks like one or capital i
```

For the conditional compilation, the work-around is duplicating the
whole containing function with a `#[cfg]`, or breaking the conditional
code into a its own function. This does mean that any variables need
to be explicitly passed as arguments.

The sort of things one could do with other arbitrary annotations are

```rust
#[allowed_unsafe_actions(ffi)]
#[audited="2014-04-22"]
unsafe { ... }
```

and then have an external tool that checks that that `unsafe` block's
only unsafe actions are FFI, or a tool that lists blocks that have
been changed since the last audit or haven't been audited ever.

The minimum useful functionality would be supporting attributes on
blocks and `let` statements, since these are flexible enough to allow
for relatively precise attribute handling.

# Detailed design

Normal attribute syntax on `let` statements and blocks.

```rust
fn foo() {
    #[attr1]
    let x = 1;

    #[attr2]
    {
        // code
    }

    #[attr3]
    unsafe {
        // code
    }
}
```

## Extension to arbitrary expressions

It would also be theoretically possible to extend this to support
arbitrary expressions (rather than just blocks, which are themselves
expressions). This would allow writing

```rust
fn foo() {
    #[attr4] foo();

    #[attr5] if cond {
        bar()
    } else #[attr6] {
        baz()
    }

    let x = #[attr7] 1;

    qux(3 + #[attr8] 2);

    foo(x, #[attr9] y, z);
}
```

These last examples indicate a possible difficulty: what happens with
say `1 + #[cfg(foo)] 2`? This should be an error, i.e. `#[cfg]` is
only allowed on the "exterior" of expressions, that is, legal in all
examples above except `#[attr7]` and `#[attr8]`. `#[attr9]` is
questionable: if it were a `cfg`, then it could reasonably be
interpreted as meaning `foo(x, z)` or `foo(x, y, z)` conditional on
the `cfg`.

Allowing attributes there would also require considering
precedence. There are two sensible options, `#[...]` binds tighter
than everything else, i.e. `#[attr] 1 + 2` is `(#[attr] 1) + 2`, or it
is weaker, so `#[attr] 1 + 2` is `#[attr] (1 + 2)`.

# Alternatives

These instances could possibly be approximated with macros and helper
functions, but to a low degree degree (e.g. how would one annotate a
general `unsafe` block).

# Unresolved questions

- Are the complications of allowing attributes on arbitrary
  expressions worth the benefits?

- Which precedence should attributes have on arbitrary expressions?
