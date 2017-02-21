# Identifiers

An identifier is any nonempty Unicode[^non_ascii_idents] string of the following form:

Either

   * The first character has property `XID_start`
   * The remaining characters have property `XID_continue`

Or

   * The first character is `_`
   * The identifier is more than one character, `_` alone is not an identifier
   * The remaining characters have property `XID_continue`

that does _not_ occur in the set of [keywords].

> **Note**: `XID_start` and `XID_continue` as character properties cover the
> character ranges used to form the more familiar C and Java language-family
> identifiers.

[keywords]: ../grammar.html#keywords
[^non_ascii_idents]: Non-ASCII characters in identifiers are currently feature
  gated. This is expected to improve soon.
