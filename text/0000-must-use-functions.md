- Feature Name: none?
- Start Date: 2015-02-18
- RFC PR: [rust-lang/rfcs#886](https://github.com/rust-lang/rfcs/pull/886)
- Rust Issue: (leave this empty)

# Summary

Support the `#[must_use]` attribute on arbitrary functions, to make
the compiler lint when a call to such a function is ignored. Mark
`Result::{ok, err}` `#[must_use]`.

# Motivation

The `#[must_use]` lint is extremely useful for ensuring that values
that are likely to be important are handled, even if by just
explicitly ignoring them with, e.g., `let _ = ...;`. This expresses
the programmers intention clearly, so that there is less confusion
about whether, for example, ignoring the possible error from a `write`
call is intentional or just an accidental oversight.

Rust has got a lot of mileage out connecting the `#[must_use]` lint to
specific types: types like `Result`, `MutexGuard` (any guard, in
general) and the lazy iterator adapters have narrow enough use cases
that the programmer usually wants to do something with them. These
types are marked `#[must_use]` and the compiler will print an error if
a semicolon ever throws away a value of that type:

```rust
fn returns_result() -> Result<(), ()> {
    Ok(())
}

fn ignore_it() {
    returns_result();
}
```

```
test.rs:6:5: 6:11 warning: unused result which must be used, #[warn(unused_must_use)] on by default
test.rs:6     returns_result();
              ^~~~~~~~~~~~~~~~~
```

However, not every "important" (or, "usually want to use") result can
be a type that can be marked `#[must_use]`, for example, sometimes
functions return unopinionated type like `Option<...>` or `u8` that
may lead to confusion if they are ignored. For example, the `Result<T,
E>` type provides

```rust
pub fn ok(self) -> Option<T> {
    match self {
        Ok(x)  => Some(x),
        Err(_) => None,
    }
}
```

to view any data in the `Ok` variant as an `Option`. Notably, this
does no meaningful computation, in particular, it does not *enforce*
that the `Result` is `ok()`. Someone reading a line of code
`returns_result().ok();` where the returned value is unused
cannot easily tell if that behaviour was correct, or if something else
was intended, possibilities include:

- `let _ = returns_result();` to ignore the result (as
  `returns_result().ok();` does),
- `returns_result().unwrap();` to panic if there was an error,
- `returns_result().ok().something_else();` to do more computation.

This is somewhat problematic in the context of `Result` in particular,
because `.ok()` does not really (in the authors opinion) represent a
meaningful use of the `Result`, but it still silences the
`#[must_use]` error.

These cases can be addressed by allowing specific functions to
explicitly opt-in to also having important results, e.g. `#[must_use]
fn ok(self) -> Option<T>`. This is a natural generalisation of
`#[must_use]` to allow fine-grained control of context sensitive info.

# Detailed design

If a semicolon discards the result of a function or method tagged with
`#[must_use]`, the compiler will emit a lint message (under same lint
as `#[must_use]` on types). An optional message `#[must_use = "..."]`
will be printed, to provide the user with more guidance.

```rust
#[must_use]
fn foo() -> u8 { 0 }


struct Bar;

impl Bar {
     #[must_use = "maybe you meant something else"]
     fn baz(&self) -> Option<String> { None }
}

fn qux() {
    foo(); // warning: unused result that must be used
    Bar.baz(); // warning: unused result that must be used: maybe you meant something else
}
```


# Drawbacks

This adds a little more complexity to the `#[must_use]` system, and
may be misused by library authors (but then, many features may be
misused).

The rule stated doesn't cover every instance where a `#[must_use]`
function is ignored, e.g. `(foo());` and `{ ...; foo() };` will not be
picked up, even though it is passing the result through a piece of
no-op syntax. This could be tweaked. Notably, the type-based rule doesn't
have this problem, since that sort of "passing-through" causes the
outer piece of syntax to be of the `#[must_use]` type, and so is
considered for the lint itself.

`Result::ok` is occasionally used for silencing the `#[must_use]`
error of `Result`, i.e. the ignoring of `foo().ok();` is
intentional. However, the most common way do ignore such things is
with `let _ =`, and `ok()` is rarely used in comparison, in most
code-bases: 2 instances in the rust-lang/rust codebase (vs. nearly 400
text matches for `let _ =`) and 4 in the servo/servo (vs. 55 `let _
=`). See the appendix for a more formal treatment of this
question. Yet another way to write this is `drop(foo())`, although
neither this nor `let _ =` have the method chaining style.

Marking functions `#[must_use]` is a breaking change in certain cases,
e.g. if someone is ignoring their result and has the relevant lint (or
warnings in general) set to be an error. This is a general problem of
improving/expanding lints.

# Alternatives

- Adjust the rule to propagate `#[must_used]`ness through parentheses
  and blocks, so that `(foo());`, `{ foo() };` and even `if cond {
  foo() } else { 0 };` are linted.

- Provide an additional method on `Result`, e.g. `fn ignore(self) {}`, so
  that users who wish to ignore `Result`s can do so in the method
  chaining style: `foo().ignore();`.

# Unresolved questions

- Are there many other functions in the standard library/compiler
  would benefit from `#[must_use]`?
- Should this be feature gated?

# Appendix: is this going to affect "most code-bases"?

(tl;dr: unlikely.)

@mahkoh stated:

> -1. I, and most code-bases, use ok() to ignore Result.

Let's investigate.

I sampled 50 random projects on [Rust CI](http://rust-ci.org), and
grepped for `\.ok` and `let _ =`.

## Methodology

Initially just I scrolled around and clicked things, may 10-15, the
rest were running this JS `var list = $("a");
window.open(list[(Math.random() * list.length) | 0].href, '_blank')`
to open literally random links in a new window. Links that were not
projects (including 404s from deleted projects) and duplicates were
ignored. The grepping was performed by running `runit url`, where
`runit` is the shell function:

```bash
function runit () { cd ~/tmp; git clone $1; cd $(basename $1); git grep '\.ok' | wc -l; git grep 'let _ =' | wc -l; }
```

If there were any `ok`s, I manually read the grep to see if they were
used on not.

## Data

| repo | used `\.ok` | unused `\.ok` | `let _ =` |
|------|-------------|---------------|-----------|
| https://github.com/csherratt/obj | 9 | 0 | 1 |
| https://github.com/csherratt/snowmew | 16 | 0 | 0 |
| https://github.com/bluss/petulant-avenger-graphlibrary | 0 | 0 | 12 |
| https://github.com/uutils/coreutils | 15 | 0 | 1 |
| https://github.com/apoelstra/rust-bitcoin/ | 5 | 0 | 3 |
| https://github.com/emk/abort_on_panic-rs | 0 | 0 | 1 |
| https://github.com/japaric/parallel.rs | 2 | 0 | 0 |
| https://github.com/phildawes/racer | 15 | 0 | 0 |
| https://github.com/zargony/rust-fuse | 7 | 7 | 0 |
| https://github.com/jakub-/rust-instrumentation | 0 | 0 | 2 |
| https://github.com/andelf/rust-iconv | 14 | 0 | 0 |
| https://github.com/pshc/brainrust | 25 | 0 | 0 |
| https://github.com/andelf/rust-2048 | 3 | 0 | 0 |
| https://github.com/PistonDevelopers/vecmath | 0 | 0 | 2 |
| https://github.com/japaric/serial.rs | 1 | 0 | 0 |
| https://github.com/servo/html5ever | 14 | 0 | 1 |
| https://github.com/sfackler/r2d2 | 8 | 0 | 0 |
| https://github.com/jamesrhurst/rust-metaflac | 2 | 0 | 0 |
| https://github.com/arjantop/rust-bencode | 3 | 0 | 1 |
| https://github.com/Azdle/dolos | 0 | 2 | 0 |
| https://github.com/ogham/exa | 2 | 0 | 0 |
| https://github.com/aatxe/irc-services | 0 | 0 | 5 |
| https://github.com/nwin/chatIRC | 0 | 0 | 8 |
| https://github.com/reima/rustboy | 1 | 0 | 2 |

These had no matches at all for `.ok` or `let _ =`:

- https://github.com/hjr3/hal-rs,
- https://github.com/KokaKiwi/lua-rs,
- https://github.com/dwrensha/capnpc-rust,
- https://github.com/samdoshi/portmidi-rs,
- https://github.com/PistonDevelopers/graphics,
- https://github.com/vberger/ircc-rs,
- https://github.com/stainless-steel/temperature,
- https://github.com/chris-morgan/phantom-enum,
- https://github.com/jeremyletang/rust-portaudio,
- https://github.com/tikue/rust-ml,
- https://github.com/FranklinChen/rust-tau,
- https://github.com/GuillaumeGomez/rust-GSL,
- https://github.com/andelf/rust-httpc,
- https://github.com/huonw/stable_vec,
- https://github.com/TyOverby/rust-termbox,
- https://github.com/japaric/stats.rs,
- https://github.com/omasanori/mesquite,
- https://github.com/andelf/rust-iconv,
- https://github.com/aatxe/dnd,
- https://github.com/pshc/brainrust,
- https://github.com/vsv/rustulator,
- https://github.com/erickt/rust-mongrel2,
- https://github.com/Geal/rust-csv,
- https://github.com/vhbit/base32-rs,
- https://github.com/PistonDevelopers/event,
- https://github.com/untitaker/rust-atomicwrites.

Disclosure, `snowmew` and `coreutils` were explicitly selected after
recognising their names (i.e. non-randomly), but this before the
`runit` script was used, and before any grepping was performed in any
of these projects.

The data in R form if you wish to play with it yourself:
```r
structure(list(used.ok = c(9, 16, 0, 15, 5, 0, 2, 15, 7, 0, 14,
25, 3, 0, 1, 14, 8, 2, 3, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), unused.ok = c(0,
0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0), let = c(1, 0, 12, 1, 3, 1, 0, 0, 0, 2,
0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 5, 8, 2, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), .Names = c("used.ok",
"unused.ok", "let"), row.names = c(NA, -50L), class = "data.frame")
```

## Analysis

I will assume that a crate author uses *either* `let _ =` or `\.ok()`
for ignoring `Result`s, but not both. The crates with neither `let _
=`s nor unused `.ok()`s are not interesting, as they haven't indicated
a preference either way. Removing those leaves 14 crates, 2 of which
use `\.ok()` and 12 of which use `let _ =`.

The null hypothesis is that `\.ok()` is used at least as much as `let
_ =`. A one-sided binomial test (e.g. `binom.test(c(2, 12),
alternative = "less")` in R) has p-value 0.007, leading me to reject
the null hypothesis and accept the alternative, that `let _ =` is used
more than `\.ok`.

(Sorry for the frequentist analysis.)
