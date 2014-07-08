- Start Date: 2014-03-20
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

Allow attributes on more places inside functions, such as statements,
blocks and expressions.

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

Normal attribute syntax on `let` statements, blocks and expressions.

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
    #[attr4] foo();

    #[attr5]
    if cond {
        bar()
    } else #[attr6] if cond {
        baz()
    } else #[attr7] {

    };

    let x = #[attr8] 1;

    qux(3 + #[attr9] 2);

    foo(x, #[attr10] y, z);
}
```

## `cfg`

It is an error to place a `#[cfg]` attribute on a non-statement
expressions, including `if`s/blocks inside an `if`/`else` chain, that
is, `attr1`--`attr7` can legally be `#[cfg(foo)]`, but
`attr8`--`attr10` cannot, since it makes little sense to strip code
down to `let x = ;`.

Attributes bind tighter than any operator, that is `#[attr] x op y` is
always parsed as `(#[attr] x) op y`.

## Inner attributes

Inner attributes can be placed at the top of blocks (and other
structure incorporating a block) and apply to that block.

```rust
{
    #![attr11]

    foo()
}

match bar {
    #![attr12]

    _ => {}
}

if cond {
    #![attr13]
}

// are the same as

#[attr11]
{
    foo()
}

#[attr12]
match bar {
    _ => {}
}

#[attr13]
if cond {
}
```


# Alternatives

These instances could possibly be approximated with macros and helper
functions, but to a low degree degree (e.g. how would one annotate a
general `unsafe` block).

Only allowing attributes on "statement expressions" that is,
expressions at the top level of a block,

# Unresolved questions

Are the complications of allowing attributes on arbitrary
expressions worth the benefits?

The interaction with `if`/`else` chains are somewhat subtle, and it
may be worth introducing "interior" and "exterior" attributes on `if`, or
just disallowing them entirely.

```rust
#[cfg(not(foo))]
if cond1 {
} else #[cfg(not(bar))] if cond2 {
} else #[cfg(not(baz))] {
}
```

- `--cfg foo`: could be either removing the whole chain ("exterior") or
  equivalent to `if cond2 {} else {}` ("interior").
- `--cfg bar`: could be either `if cond1 {}` or `if cond1 {} else {}`
- `--cfg baz`: equivalent to `if cond1 {} else if cond2 {}` (no subtlety).
- `--cfg foo --cfg bar`: could be removing the whole chain or the two
  `if` branches (leaving only the `else` branch).

This can be addressed by having `#[attr] if cond { ...` be an exterior
attribute (applying to the whole `if`/`else` chain) and
`if cond #[attr] { ... ` be an interior attribute (applying to only
the current `if` branch).
