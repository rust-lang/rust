- Start Date: 2014-03-20
- RFC PR: [rust-lang/rfcs#16](https://github.com/rust-lang/rfcs/pull/16)
- Rust Issue: [rust-lang/rust#15701](https://github.com/rust-lang/rust/issues/15701)

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

    let x = #[attr5] 1;

    qux(3 + #[attr6] 2);

    foo(x, #[attr7] y, z);
}
```

Attributes bind tighter than any operator, that is `#[attr] x op y` is
always parsed as `(#[attr] x) op y`.

## `cfg`

It is definitely an error to place a `#[cfg]` attribute on a
non-statement expressions, that is, `attr1`--`attr4` can possibly be
`#[cfg(foo)]`, but `attr5`--`attr7` cannot, since it makes little
sense to strip code down to `let x = ;`.

However, like `#ifdef` in C/C++, widespread use of `#[cfg]` may be an
antipattern that makes code harder to read. This RFC is just adding
the ability for attributes to be placed in specific places, it is not
mandating that `#[cfg]` actually be stripped in those places (although
it should be an error if it is ignored).

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

// are the same as

#[attr11]
{
    foo()
}

#[attr12]
match bar {
    _ => {}
}
```

## `if`

Attributes would be disallowed on `if` for now, because the
interaction with `if`/`else` chains are funky, and can be simulated in
other ways.

```rust
#[cfg(not(foo))]
if cond1 {
} else #[cfg(not(bar))] if cond2 {
} else #[cfg(not(baz))] {
}
```

There is two possible interpretations of such a piece of code,
depending on if one regards the attributes as attaching to the whole
`if ... else` chain ("exterior") or just to the branch on which they
are placed ("interior").

- `--cfg foo`: could be either removing the whole chain (exterior) or
  equivalent to `if cond2 {} else {}` (interior).
- `--cfg bar`: could be either `if cond1 {}` (*e*) or `if cond1 {}
  else {}` (*i*)
- `--cfg baz`: equivalent to `if cond1 {} else if cond2 {}` (no subtlety).
- `--cfg foo --cfg bar`: could be removing the whole chain (*e*) or the two
  `if` branches (leaving only the `else` branch) (*i*).

(This applies to any attribute that has some sense of scoping, not
just `#[cfg]`, e.g. `#[allow]` and `#[warn]` for lints.)

As such, to avoid confusion, attributes would not be supported on
`if`. Alternatives include using blocks:

```rust
#[attr] if cond { ... } else ...
// becomes, for an exterior attribute,
#[attr] {
    if cond { ... } else ...
}
// and, for an interior attribute,
if cond {
    #[attr] { ... }
} else ...
```

And, if the attributes are meant to be associated with the actual
branching (e.g. a hypothetical `#[cold]` attribute that indicates a
branch is unlikely), one can annotate `match` arms:

```rust
match cond {
    #[attr] true => { ... }
    #[attr] false => { ... }
}
```

# Drawbacks

This starts mixing attributes with nearly arbitrary code, possibly
dramatically restricting syntactic changes related to them, for
example, there was some consideration for using `@` for attributes,
this change may make this impossible (especially if `@` gets reused
for something else, e.g. Python is
[using it for matrix multiplication](http://legacy.python.org/dev/peps/pep-0465/)). It
may also make it impossible to use `#` for other things.

As stated above, allowing `#[cfg]`s everywhere can make code harder to
reason about, but (also stated), this RFC is not for making such
`#[cfg]`s be obeyed, it just opens the language syntax to possibly
allow it.

# Alternatives

These instances could possibly be approximated with macros and helper
functions, but to a low degree degree (e.g. how would one annotate a
general `unsafe` block).

Only allowing attributes on "statement expressions" that is,
expressions at the top level of a block, this is slightly limiting;
but we can expand to support other contexts backwards compatibly in
the future.

The `if`/`else` issue may be able to be resolved by introducing
explicit "interior" and "exterior" attributes on `if`: by having
`#[attr] if cond { ...` be an exterior attribute (applying to the
whole `if`/`else` chain) and `if cond #[attr] { ... ` be an interior
attribute (applying to only the current `if` branch). There is no
difference between interior and exterior for an `else {` branch, and
so `else #[attr] {` is sufficient.


# Unresolved questions

Are the complications of allowing attributes on arbitrary
expressions worth the benefits?
