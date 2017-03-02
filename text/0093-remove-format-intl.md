- Start Date: 2014-06-10
- RFC PR: [rust-lang/rfcs#93](https://github.com/rust-lang/rfcs/pull/93)
- Rust Issue: [rust-lang/rust#14812](https://github.com/rust-lang/rust/issues/14812)

# Summary

Remove localization features from format!, and change the set of escapes
accepted by format strings. The `plural` and `select` methods would be removed,
`#` would no longer need to be escaped, and `{{`/`}}` would become escapes for
`{` and `}`, respectively.

# Motivation

Localization is difficult to implement correctly, and doing so will
likely not be done in the standard library, but rather in an external library.
After talking with others much more familiar with localization, it has
come to light that our ad-hoc "localization support" in our format
strings are woefully inadequate for most real use cases of support for
localization.

Instead of having a half-baked unused system adding complexity to the compiler
and libraries, the support for this functionality would be removed from format
strings.

# Detailed design

The primary localization features that `format!` supports today are the
`plural` and `select` methods inside of format strings. These methods are
choices made at format-time based on the input arguments of how to format a
string. This functionality would be removed from the compiler entirely.

As fallout of this change, the `#` special character, a back reference to the
argument being formatted, would no longer be necessary. In that case, this
character no longer needs an escape sequence.

The new grammar for format strings would be as follows:

```
format_string := <text> [ format <text> ] *
format := '{' [ argument ] [ ':' format_spec ] '}'
argument := integer | identifier

format_spec := [[fill]align][sign]['#'][0][width]['.' precision][type]
fill := character
align := '<' | '>'
sign := '+' | '-'
width := count
precision := count | '*'
type := identifier | ''
count := parameter | integer
parameter := integer '$'
```

The current syntax can be found at http://doc.rust-lang.org/std/fmt/#syntax to
see the diff between the two

## Choosing a new escape sequence

Upon landing, there was a significant amount of discussion about the escape
sequence that would be used in format strings. Some context can be found on some
[old pull requests][1], and the current escape mechanism has been the source of
[much confusion][2]. With the removal of localization methods, and
namely nested format directives, it is possible to reconsider the choices of
escaping again.

[1]: https://github.com/mozilla/rust/pull/9161
[2]: https://github.com/mozilla/rust/issues/12814

The only two characters that need escaping in format strings are `{` and `}`.
One of the more appealing syntaxes for escaping was to double the character to
represent the character itself. This would mean that `{{` is an escape for a `{`
character, while `}}` would be an escape for a `}` character.

Adopting this scheme would avoid clashing with Rust's string literal escapes.
There would be no "double escape" problem. More details on this can be found in
the comments of an [old PR][1].

# Drawbacks

The localization methods of select/plural are genuinely used for
applications that do not involve localization. For example, the compiler
and rustdoc often use plural to easily create plural messages. Removing this
functionality from format strings would impose a burden of likely dynamically
allocating a string at runtime or defining two separate format strings.

Additionally, changing the syntax of format strings is quite an invasive change.
Raw string literals serve as a good use case for format strings that must escape
the `{` and `}` characters. The current system is arguably good enough to pass
with for today.

# Alternatives

The major localization approach explored has been l20n, which has shown
itself to be fairly incompatible with the way format strings work today.
Different localization systems, however, have not been explored. Systems
such as gettext would be able to leverage format strings quite well, but it
was claimed that gettext for localization is inadequate for modern
use-cases.

It is also an unexplored possibility whether the current format string syntax
could be leveraged by l20n. It is unlikely that time will be allocated to polish
off an localization library before 1.0, and it is currently seen as
undesirable to have a half-baked system in the libraries rather than a
first-class well designed system.

# Unresolved questions

* Should localization support be left in `std::fmt` as a "poor man's"
  implementation for those to use as they see fit?
