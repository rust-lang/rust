- Start Date: 2014-11-05
- RFC PR: https://github.com/rust-lang/rfcs/pull/446
- Rust Issue: https://github.com/rust-lang/rust/issues/19739

# Summary

Remove `\u203D` and `\U0001F4A9` unicode string escapes, and add
[ECMAScript 6-style](https://mathiasbynens.be/notes/javascript-escapes#unicode-code-point)
`\u{1F4A9}` escapes instead.

# Motivation

The syntax of `\u` followed by four hexadecimal digits dates from when Unicode
was a 16-bit encoding, and only went up to U+FFFF.
`\U` followed by eight hex digits was added as a band-aid
when Unicode was extended to U+10FFFF,
but neither four nor eight digits particularly make sense now.

Having two different syntaxes with the same meaning but that apply
to different ranges of values is inconsistent and arbitrary.
This proposal unifies them into a single syntax that has a precedent
in ECMAScript a.k.a. JavaScript.


# Detailed design

In terms of the grammar in [The Rust Reference](
http://doc.rust-lang.org/reference.html#character-and-string-literals),
replace:

```
unicode_escape : 'u' hex_digit 4
               | 'U' hex_digit 8 ;
```

with

```
unicode_escape : 'u' '{' hex_digit+ 6 '}'
```

That is, `\u{` followed by one to six hexadecimal digits, followed by `}`.

The behavior would otherwise be identical.

## Migration strategy

In order to provide a graceful transition from the old `\uDDDD` and
`\UDDDDDDDD` syntax to the new `\u{DDDDD}` syntax, this feature
should be added in stages:

* Stage 1: Add support for the new `\u{DDDDD}` syntax, without removing
previous support for `\uDDDD` and `\UDDDDDDDD`.

* Stage 2: Warn on occurrences of `\uDDDD` and `\UDDDDDDDD`. Convert
all library code to use `\u{DDDDD}` instead of the old syntax.

* Stage 3: Remove support for the old syntax entirely (preferably
during a separate release from the one that added the warning from
Stage 2).

# Drawbacks

* This is a breaking change and updating code for it manually is annoying.
  It is however very mechanical, and we could provide scripts to automate it.
* Formatting templates already use curly braces.
  Having multiple curly braces pairs in the same strings that have a very
  different meaning can be surprising:
  `format!("\u{e8}_{e8}", e8 = "é")` would be `"è_é"`.
  However, there is a precedent of overriding characters:
  `\` can start an escape sequence both in the Rust lexer for strings
  and in regular expressions.


# Alternatives

* Status quo: don’t change the escaping syntax.
* Add the new `\u{…}` syntax, but also keep the existing `\u` and `\U` syntax.
  This is what ES 6 does, but only to keep compatibility with ES 5.
  We don’t have that constaint pre-1.0.

# Unresolved questions

None so far.
