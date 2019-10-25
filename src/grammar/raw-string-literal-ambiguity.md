Rust's lexical grammar is not context-free. Raw string literals are the source
of the problem. Informally, a raw string literal is an `r`, followed by `N`
hashes (where N can be zero), a quote, any characters, then a quote followed
by `N` hashes. Critically, once inside the first pair of quotes,
another quote cannot be followed by `N` consecutive hashes. e.g.
`r###""###"###` is invalid.

This grammar describes this as best possible:

    R -> 'r' S
    S -> '"' B '"'
    S -> '#' S '#'
    B -> . B
    B -> ε

Where `.` represents any character, and `ε` the empty string. Consider the
string `r#""#"#`. This string is not a valid raw string literal, but can be
accepted as one by the above grammar, using the derivation:

    R : #""#"#
    S : ""#"
    S : "#
    B : #
    B : ε

(Where `T : U` means the rule `T` is applied, and `U` is the remainder of the
string.) The difficulty arises from the fact that it is fundamentally
context-sensitive. In particular, the context needed is the number of hashes.

To prove that Rust's string literals are not context-free, we will use
the fact that context-free languages are closed under intersection with
regular languages, and the
[pumping lemma for context-free languages](https://en.wikipedia.org/wiki/Pumping_lemma_for_context-free_languages).

Consider the regular language `R = r#+""#*"#+`. If Rust's raw string literals are
context-free, then their intersection with `R`, `R'`, should also be context-free.
Therefore, to prove that raw string literals are not context-free,
it is sufficient to prove that `R'` is not context-free.

The language `R'` is `{r#^n""#^m"#^n | m < n}`.

Assume `R'` *is* context-free. Then `R'` has some pumping length `p > 0` for which
the pumping lemma applies. Consider the following string `s` in `R'`:

`r#^p""#^{p-1}"#^p`

e.g. for `p = 2`: `s = r##""#"##`

Then `s = uvwxy` for some choice of `uvwxy` such that `vx` is non-empty,
`|vwx| < p+1`, and `uv^iwx^iy` is in `R'` for all `i >= 0`.

Neither `v` nor `x` can contain a `"` or `r`, as the number of these characters
in any string in `R'` is fixed. So `v` and `x` contain only hashes.
Consequently, of the three sequences of hashes, `v` and `x` combined
can only pump two of them.
If we ever choose the central sequence of hashes, then one of the outer sequences
will not grow when we pump, leading to an imbalance between the outer sequences.
Therefore, we must pump both outer sequences of hashes. However,
there are `p+2` characters between these two sequences of hashes, and `|vwx|` must
be less than `p+1`. Therefore we have a contradiction, and `R'` must not be
context-free.

Since `R'` is not context-free, it follows that the Rust's raw string literals
must not be context-free.
