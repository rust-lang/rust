- Feature Name: `raw_identifiers`
- Start Date: 2017-09-14
- RFC PR: [rust-lang/rfcs#2151](https://github.com/rust-lang/rfcs/pull/2151)
- Rust Issue: [rust-lang/rust#48589](https://github.com/rust-lang/rust/issues/48589)

# Summary
[summary]: #summary

Add a raw identifier format `r#ident`, so crates written in future language
epochs/versions can still use an older API that overlaps with new keywords.

# Motivation
[motivation]: #motivation

One of the primary examples of breaking changes in the epoch RFC is to add new
keywords, and specifically `catch` is the first candidate. However, since
that's seeking crate compatibility across epochs, this would leave a crate in a
newer epoch unable to use `catch` identifiers in the API of a crate in an older
epoch.  [@matklad found] 28 crates using `catch` identifiers, some public.

A raw syntax that's *always* an identifier would allow these to remain
compatible, so one can write `r#catch` where `catch`-as-identifier is needed.

[@matklad found]: https://internals.rust-lang.org/t/pre-rfc-raw-identifiers/5502/40

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Although some identifiers are reserved by the Rust language as keywords, it is
still possible to write them as raw identifiers using the `r#` prefix, like
`r#ident`.  When written this way, it will *always* be treated as a plain
identifier equivalent to a bare `ident` name, never as a keyword.

For instance, the following is an erroneous use of the `match` keyword:

```rust
fn match(needle: &str, haystack: &str) -> bool {
    haystack.contains(needle)
}
```

```text
error: expected identifier, found keyword `match`
 --> src/lib.rs:1:4
  |
1 | fn match(needle: &str, haystack: &str) -> bool {
  |    ^^^^^
```

It can instead be written as `fn r#match(needle: &str, haystack: &str)`, using
the `r#match` raw identifier, and the compiler will accept this as a true
`match` function.

Generally when defining items, you should just avoid keywords altogether and
choose a different name.  Raw identifiers require the `r#` prefix every time
they are mentioned, making them cumbersome to both the developer and users.
Usually an alternate is preferable: `crate` -> `krate`, `const` -> `constant`,
etc.

However, new Rust epochs may add to the list of reserved keywords, making a
formerly legal identifier now interpreted otherwise.  Since compatibility is
maintained between crates of different epochs, this could mean that code written
in a new epoch might not be able to name an identifier in the API of another
crate.  Using a raw identifier, it can still be named and used.

```rust
//! baseball.rs in epoch 2015
pub struct Ball;
pub struct Player;
impl Player {
    pub fn throw(&mut self) -> Result<Ball> { ... }
    pub fn catch(&mut self, ball: Ball) -> Result<()> { ... }
}
```

```rust
//! main.rs in epoch 2018 -- `catch` is now a keyword!
use baseball::*;
fn main() {
    let mut player = Player;
    let ball = player.throw()?;
    player.r#catch(ball)?;
}
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The syntax for identifiers allows an optional `r#` prefix for a raw identifier,
otherwise following the normal identifer rules.  Raw identifiers are always
interpreted as plain identifiers and never as keywords, regardless of context.
They are also treated equivalent to an identifier that wasn't raw -- for
instance, it's perfectly legal to write:

```rust
let foo = 123;
let bar = r#foo * 2;
```

# Drawbacks
[drawbacks]: #drawbacks

- New syntax is always scary/noisy/etc.
- It might not be intuitively "raw" to a user coming upon this the first time.

# Rationale and Alternatives
[alternatives]: #alternatives

If we don't have any way to refer to identifiers that were legal in prior
epochs, but later became keywords, then this may hurt interoperability between
crates of different epochs.  The `r#ident` syntax enables interoperability, and
will hopefully invoke some intuition of being raw, similar to raw strings.

The `br#ident` syntax is also possible, but I see no advantage over `r#ident`.
Identifiers don't need the same kind of distinction as `str` and `[u8]`.

A small possible alternative is to also terminate it like `r#ident#`, which
could allow non-identifier characters to be part of a raw identifier.  This
could take a cue from raw strings and allow repetition for internal `#`, like
`r##my #1 ident##`.  That doesn't allow a leading `#` or `"` though.

A different possibility is to use backticks for a string-like `` `ident` ``,
like [Kotlin], [Scala], and [Swift].  If it allows non-identifier chars, it
could embrace escapes like `\u`, and have a raw-string-identifier ``
r`slash\ident` `` and even `` r#`tick`ident`# ``.  However, backtick identifiers
are annoying to write in markdown. (e.g. ``` `` `ident` `` ```)

Backslashes could connote escaping identifiers, like `\ident`, perhaps
surrounded like `\ident\`, `\{ident}`, etc.  However, the infix RFC #1579
currently seems to be leaning towards `\op` syntax already.

Alternatives which already start legal tokens, like [C#]'s `@ident`, [Dart]'s
`#ident`, or alternate prefixes like `identifier#catch`, all break Macros 1.0
as [@kennytm demonstrated]:

```
macro_rules! x {
    (@ $a:ident) => {};
    (# $a:ident) => {};
    ($a:ident # $b:ident) => {};
    ($a:ident) => { should error };
}
x!(@catch);
x!(#catch);
x!(identifier#catch);
x!(keyword#catch);
```

C# allows Unicode escapes directly in identifiers, which also separates them
from keywords, so both `@catch` and `cl\u0061ss` are valid `class` identifiers.
Java also allows Unicode escapes, but they don't avoid keywords.

For some new keywords, there may be contextual mitigations. In the case of
`catch`, it couldn't be a fully contextual keyword because `catch { ... }` could
be a struct literal. That context might be worked around with a path, like
`old_epoch::catch { ... }` to use an identifier instead. Contexts that don't
make sense for a `catch` expression can just be identifiers, like `foo.catch()`.
However, this might not be possible for all future keywords.

There might also be a need for raw keywords in the other direction, e.g. so the
older epoch can still use the new `catch` functionality somehow. I think this
particular case is already served well enough by `do catch { ... }`, if we
choose to stabilize it that way.  Perhaps `br#keyword` could be used for this,
but that may not be a good intuitive relationship.

[C#]: https://msdn.microsoft.com/en-us/library/aa664670(v=vs.71).aspx
[Dart]: https://www.dartlang.org/guides/language/language-tour#symbols
[Kotlin]: https://kotlinlang.org/docs/reference/grammar.html
[Scala]: https://www.scala-lang.org/files/archive/spec/2.13/01-lexical-syntax.html#identifiers
[Swift]: https://developer.apple.com/library/content/documentation/Swift/Conceptual/Swift_Programming_Language/LexicalStructure.html
[@kennytm demonstrated]: https://internals.rust-lang.org/t/pre-rfc-raw-identifiers/5502/28

# Unresolved questions
[unresolved]: #unresolved-questions

- Do macros need any special care with such identifier tokens?
- Should diagnostics use the `r#` syntax when printing identifiers that overlap keywords?
- Does rustdoc need to use the `r#` syntax? e.g. to document `pub use old_epoch::*`
