- Feature Name: `line_endings`
- Start Date: 2015-07-10
- RFC PR: [rust-lang/rfcs#1212](https://github.com/rust-lang/rfcs/pull/1212)
- Rust Issue: [rust-lang/rust#28032](https://github.com/rust-lang/rust/issues/28032)

# Summary

Change all functions dealing with reading "lines" to treat both '\n' and '\r\n'
as a valid line-ending.

# Motivation

The current behavior of these functions is to treat only '\n' as line-ending.
This is surprising for programmers experienced in other languages. Many
languages open files in a "text-mode" per default, which means when they iterate
over the lines, they don't have to worry about the two kinds of line-endings.
Such programmers will be surprised to learn that they have to take care of such
details themselves in Rust. Some may not even have heard of the distinction
between two styles of line-endings.

The current design also violates the "do what I mean" principle. Both '\r\n' and
'\n' are widely used as line-separators. By talking about the concept of
"lines", it is clear that the current file (or buffer, really) is considered to
be in text format. It is thus very reasonable to expect "lines" to apply to both
kinds of encoding lines in binary format.

In particular, if the crate is developed on Linux or Mac, the programmer will
probably have most of his input encoded with only '\n' for the line-endings. He
may use the functions talking about "lines", and they will work all right. It is
only when someone runs this crate on input that contains '\r\n' that the bug
will be uncovered. The editor has personally run into this issue when reading
line-by-line from stdin, with the program suddenly failing on Windows.

# Detailed design

The following functions will have to be changed: `BufRead::lines` and
`str::lines`. They both should treat '\r\n' as marking the end of a line. This
can be implemented, for example, by first splitting at '\n' like now and then
removing a trailing '\r' right before returning data to the caller.

Furthermore, `str::lines_any` (the only function currently dealing with both
kinds of line-endings) is deprecated, as it is then functionally equivalent with
`str::lines`.

# Drawbacks

This is a semantics-breaking change, changing the behavior of released, stable
API. However, as argued above, the new behavior is much less surprising than the
old one - so one could consider this fixing a bug in the original
implementation. There are alternatives available for the case that one really
wants to split at '\n' only, namely `BufRead::split` and `str::split`. However,
`BufRead:split` does not iterate over `String`, but rather over `Vec<u8>`, so
users have to insert an additional explicit call to `String::from_utf8`.

# Alternatives

There's the obvious alternative of not doing anything. This leaves a gap in the
features Rust provides to deal with text files, making it hard to treat both
kinds of line-endings uniformly.

The second alternative is to add `BufRead::lines_any` which works similar to
`str::lines_any` in that it deals with both '\n' and '\r\n'. This provides all
the necessary functionality, but it still leaves people with the need to choose
one of the two functions - and potentially choosing the wrong one. In
particular, the functions with the shorter, nicer name (the existing ones) will
almost always *not* be the right choice.

# Unresolved questions

None I can think of.
