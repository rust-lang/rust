- Feature Name: str-words
- Start Date: 2015-04-10
- RFC PR: [rust-lang/rfcs#1054](https://github.com/rust-lang/rfcs/pull/1054)
- Rust Issue: [rust-lang/rust#24543](https://github.com/rust-lang/rust/issues/24543)

# Summary

Rename or replace `str::words` to side-step the ambiguity of “a word”.


# Motivation

The [`str::words`](http://doc.rust-lang.org/std/primitive.str.html#method.words) method
is currently marked `#[unstable(reason = "the precise algorithm to use is unclear")]`.
Indeed, the concept of “a word” is not easy to define in presence of punctuation
or languages with various conventions, including not using spaces at all to separate words.

[Issue #15628](https://github.com/rust-lang/rust/issues/15628) suggests
changing the algorithm to be based on [the *Word Boundaries* section of
*Unicode Standard Annex #29: Unicode Text Segmentation*](http://www.unicode.org/reports/tr29/#Word_Boundaries).

While a Rust implementation of UAX#29 would be useful, it belong on crates.io more than in `std`:

* It carries significant complexity that may be surprising from something that looks as simple
  as a parameter-less “words” method in the standard library.
  Users may not be aware of how subtle defining “a word” can be.
* It is not a definitive answer. The standard itself notes:

  > It is not possible to provide a uniform set of rules that resolves all issues across languages
  > or that handles all ambiguous situations within a given language.
  > The goal for the specification presented in this annex is to provide a workable default;
  > tailored implementations can be more sophisticated.

  and gives many examples of such ambiguous situations.

Therefore, `std` would be better off avoiding the question of defining word boundaries entirely.


# Detailed design

Rename the `words` method to `split_whitespace`, and keep the current behavior unchanged.
(That is, return an iterator equivalent to `s.split(char::is_whitespace).filter(|s| !s.is_empty())`.)

Rename the return type `std::str::Words` to `std::str::SplitWhitespace`.

Optionally, keep a `words` wrapper method for a while, both `#[deprecated]` and `#[unstable]`,
with an error message that suggests `split_whitespace` or the chosen alternative.


# Drawbacks

`split_whitespace` is very similar to the existing `str::split<P: Pattern>(&self, P)` method,
and having a separate method seems like weak API design. (But see below.)


# Alternatives

* Replace `str::words` with `struct Whitespace;` with a custom `Pattern` implementation,
  which can be used in `str::split`.
  However this requires the `Whitespace` symbol to be imported separately.
* Remove `str::words` entirely and tell users to use
  `s.split(char::is_whitespace).filter(|s| !s.is_empty())` instead.


# Unresolved questions

Is there a better alternative?
