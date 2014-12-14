- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

> One para explanation of the feature.

Stabilize all string functions working with search patterns around a new
generic API that provides a unfied way to define and use those patterns.

# Motivation

> Why are we doing this? What use cases does it support? What is the expected outcome?

Right now, string slices define a couple of methods for string
manipulation that work with user provided values that act as
search patterns. For example, `split()` takes an type implementing `CharEq`
to split the slice at all codepoints that match that predicate.

Among these methods, the notion of what exactly is being used as a search
pattern varies inconsistently: Many work with the generic `CharEq`,
which only looks at a single codepoint at a time; and some
work with `char` or `&str` directly, sometimes duplicating a method to
provide operations for both.

This presents a couple of issues:

- The API is inconsistent.
- The API duplicates similar operations on different types. (`contains` vs `contains_char`)
- The API does not provide all operations for all types. (No `rsplit` for `&str` patterns)
- The API is not extensible, eg to allow splitting at regex matches.
- The API offers no way to statically decide between different basic search algorithms
  for the same pattern, for example to use Boojer Moore string searching

> TODO: Spelling above

At the moment, the full set of relevant string methods roughly looks like this:

```rust
pub trait StrExt for ?Sized {
    fn contains(&self, needle: &str) -> bool;
    fn contains_char(&self, needle: char) -> bool;

    fn split<Sep: CharEq>(&self, sep: Sep) -> CharSplits<Sep>;
    fn splitn<Sep: CharEq>(&self, sep: Sep, count: uint) -> CharSplitsN<Sep>;
    fn rsplitn<Sep: CharEq>(&self, sep: Sep, count: uint) -> CharSplitsN<Sep>;
    fn split_terminator<Sep: CharEq>(&self, sep: Sep) -> CharSplits<Sep>;
    fn split_str<'a>(&'a self, &'a str) -> StrSplits<'a>;

    fn match_indices<'a>(&'a self, sep: &'a str) -> MatchIndices<'a>;

    fn starts_with(&self, needle: &str) -> bool;
    fn ends_with(&self, needle: &str) -> bool;

    fn trim_chars<C: CharEq>(&self, to_trim: C) -> &'a str;
    fn trim_left_chars<C: CharEq>(&self, to_trim: C) -> &'a str;
    fn trim_right_chars<C: CharEq>(&self, to_trim: C) -> &'a str;

    fn find<C: CharEq>(&self, search: C) -> Option<uint>;
    fn rfind<C: CharEq>(&self, search: C) -> Option<uint>;
    fn find_str(&self, &str) -> Option<uint>;

    // ...
}
```

This RFC proposes to fix those issues by providing a unified `Pattern` trait
that all "string pattern" types would implement, and that would be used by the string API
exclusively.

As an additional design goal, the new abstractions should also not pose a problem
for optimization - like for iterators, a concrete instance should produce similar
machine code to a hardcoded optimized loop written in C.

> Idea: Parallel trait hierachy not using unsafe, that will use checks

# Detailed design

> This is the bulk of the RFC. Explain the design in enough detail for somebody familiar
with the language to understand, and for somebody familiar with the compiler to implement.
This should get into specifics and corner-cases, and include examples of how the feature is used.

> Goal: A working draft with lifetimes

## New traits

First, new traits will be added to the `str` module in the std library:

```rust
trait Pattern<'a> {
    type MatcherImpl: Matcher<'a>;

    fn into_matcher(self, haystack: &'a str) -> Self::MatcherImpl;

    // Can be implemented to optimize the "find only" case.
    fn is_contained_in(self, haystack: &'a str) -> bool {
        self.into_matcher(s).next_match().is_some()
    }
}
```

A `Pattern` represents a builder for an associated type implementing a
family of `Matcher` traits (see below), and will be implemented by all types that
represent string patterns, which includes:

- `char` and `&str`
- Everything implementing `CharEq`
- Additional types like `&Regex` or `Ascii`

```rust
impl<'a>     Pattern<'a> for char       { /* ... */ }
impl<'a, 'b> Pattern<'a> for &'b str    { /* ... */ }

impl<'a, 'b> Pattern<'a> for &'b [char] { /* ... */ }
impl<'a, F>  Pattern<'a> for F where F: FnOnce(char) -> bool { /* ... */ }

impl<'a, 'b> Pattern<'a> for &'b Regex  { /* ... */ }
```

The lifetime paramter on `Pattern` exists in order to allow threading the lifetime
of the haystack (the string to be searched through) through the API, and is a workaround
for not having associated higher kinded types yet.

Consumers of this API can then call `into_matcher()` on the pattern to convert it into
a type implementing a family of `Matcher` traits:

```rust
unsafe trait Matcher<'a> {
    fn haystack(&self) -> &'a str
    fn next_match(&mut self) -> Option<(uint, uint)>;
}

unsafe trait ReverseMatcher<'a>: Matcher<'a> {
    fn next_match_back(&mut self) -> Option<(uint, uint)>;
}

trait DoubleEndedMatcher<'a>: ReverseMatcher<'a> {}
```

> TODO: Better name for the last trait

The basic idea of a `Matcher` is to expose a `Iterator`-like interface for
iterating through all matches of a pattern in the given haystack.

Similar to iterators, depending on the concrete implementation a matcher can have
additional capabilities that build on each other, which is why they will be
defined in terms of a three-tier hierachy:

- `Matcher<'a>` is the basic trait that all matchers need to implement.
  It contains a `next_match()` method that returns the `start` and `end` indices of
  the next non-overlapping match in the haystack, with the search beginning at the front
  (left) of the string. It also contains a `haystack()` getter for returning the
  actual haystack, which is the source of the `'a` lifetime on the hierarchy.
  The reason for this getter being made part of the trait is twofold:
  - Every matcher needs to store some reference to the haystack anyway.
  - Users of this trait will need access to the haystack in order
    for the individual match results to be useful.
- `ReverseMatcher<'a>` adds an `next_match_back` method, for also allowing to efficiently
  search for matches in reverse (starting from the right).
  However, the results are not required to be equal to the results of
  `next_match` in reverse, (as would be the case for the `DoubleEndedIterator` trait)
  as that can not be efficiently guaranteed for all matchers. (For an example, see further below)
- Instead `DoubleEndedMatcher<'a>` is provided as an marker trait for expressing
  that guarantee - If a matcher implements this trait, all results found from the
  left need to be equal to all results found from the right in reverse order.

As an important last detail, both
`Matcher` and `ReverseMatcher` are marked as `unsafe` traits, even though the actual methods
aren't. This is because every implementation of these traits need to ensure that all
indices returned by `next_match` and `next_match_back` lie on valid utf8 boundaries
in the used haystack.

Without that guarantee, every single match returned by a matcher would need to be
double-checked for validity, which would be unnecessary and most likely
unoptimizable work.

This is in contrast to the current hardcoded implementations, which can
make use of such guarantees because the concrete types are known
and all unsafe code needed for such optimizations is contained inside a single safe impl.

Given that most implementations of these traits will likely
live in the std library anyway, and are thoroughly tested, marking these traits `unsafe`
doesn't seem like a huge burden to bear for good, optimizable performance.

### Example for the issue with double-ended searching

Let the haystack be the string `"fooaaaaabar"`, and let the pattern be the string `"aa"`.

Then a efficient, lazy implementation of the matcher searching from the left
would find these matches:

`"foo[aa][aa]abar"`

However, the same algorithm searching from the right would find these matches:

`"fooa[aa][aa]bar"`

This discrepancy can not be avoided without additional overhead or even
allocations for caching in the reverse matcher, and thus "matching from the front" needs to
be considered a different operation than "matching from the back".

## New methods on `StrExt`

With the `Pattern` and `Matcher` traits defined and implemented, the actual `str`
methods will be changed to make use of them:

```rust
pub trait StrExt {
    fn contains<'a, P>(&'a self, pat: P) -> bool where P: Pattern<'a>;

    fn split<'a, P>(&'a self, pat: P) -> Splits<P> where P: Pattern<'a>;
    fn rsplit<'a, P>(&'a self, pat: P) -> RSplits<P> where P: Pattern<'a>;
    fn split_terminator<'a, P>(&'a self, pat: P) -> TermSplits<P> where P: Pattern<'a>;
    fn rsplit_terminator<'a, P>(&'a self, pat: P) -> RTermSplits<P> where P: Pattern<'a>;
    fn splitn<'a, P>(&'a self, pat: P, n: uint) -> NSplits<P> where P: Pattern<'a>;
    fn rsplitn<'a, P>(&'a self, pat: P, n: uint) -> RNSplits<P> where P: Pattern<'a>;

    fn matches<'a, P>(&'a self, pat: P) -> Matches<P> where P: Pattern<'a>;
    fn rmatches<'a, P>(&'a self, pat: P) -> RMatches<P> where P: Pattern<'a>;
    fn match_indices<'a, P>(&'a self, pat: P) -> MatchIndices<P> where P: Pattern<'a>;
    fn rmatch_indices<'a, P>(&'a self, pat: P) -> RMatchIndices<P> where P: Pattern<'a>;

    fn starts_with<'a, P>(&'a self, pat: P) -> bool where P: Pattern<'a>;
    fn ends_with<'a, P>(&'a self, pat: P) -> bool where P: Pattern<'a>,
                                                        P::MatcherImpl: ReverseMatcher<'a>;

    fn trim_matches<'a, P>(&'a self, pat: P) -> &'a str where P: Pattern<'a>,
                                                              P::MatcherImpl: ReverseMatcher<'a>;
    fn trim_left_matches<'a, P>(&'a self, pat: P) -> &'a str where P: Pattern<'a>;
    fn trim_right_matches<'a, P>(&'a self, pat: P) -> &'a str where P: Pattern<'a>,
                                                                    P::MatcherImpl: ReverseMatcher<'a>;

    fn find<'a, P>(&'a self, pat: P) -> Option<uint> where P: Pattern<'a>;
    fn rfind<'a, P>(&'a self, pat: P) -> Option<uint> where P: Pattern<'a>,
                                                            P::MatcherImpl: ReverseMatcher<'a>;

    // ...
}
```

These are mainly the same pattern-using methods as currently existing, only
changed to uniformly use the new pattern API. The main differences are:
- Duplicates like `contains(char)` and `contains_str(&str)` got merged into single generic methods.
- `CharEq`-centric naming got changed to `Pattern`-centric naming by changing `chars`
  to `matches` in a few method names.
- A `Matches` iterator has been added, that just returns the pattern matches as `&str` slices.
  Its uninteresting for patterns that look for a single string fragment, like the `char` and `&str`
  matcher, but useful for advanced patterns like predicates over codepoints, or regular expressions.
- All operations that can work from both the front and the back consistently exist in two versions,
  the regular front version, and a `r` prefixed reverse versions. As explained above,
  this is because both represent different operations, and thus need to be handled as such.
  To be more precise, the two can __not__ be abstracted over by providing a `DoubleEndedIterator`
  implementations, as the different results would break the requirement for double ended iterators
  to behave like a double ended queues where you just pop elements from both sides.

_However_, all iterators will still implement `DoubleEndedIterator` if the underling
matcher implements `DoubleEndedMatcher`, to keep the ability to do things like `foo.split('a').rev()`.

## Transition and deprecation plans

Most changes in this RFC can be made in such a way that code using the old hardcoded or `CharEq`-using
methods will still compile, or give deprecation warning.

It would even be possible to generically implement `Pattern` for all `CharEq` types,
making the transition more painless.

Long-term, post 1.0, it would be possible to define new sets of `Pattern` and `Matcher`
without a lifetime parameter by making use of higher kinded types in order to simplify the
string APIs. Eg, instead of `fn starts_with<'a, P>(&'a self, pat: P) -> bool where P: Pattern<'a>;`
you'd have `fn starts_with<P>(&self, pat: P) -> bool where P: Pattern;`.

In order to not break backwards-compability, these can use the same generic-impl trick to
forward to the old traits, which would roughly look like this:

```rust
unsafe trait NewPattern {
    type MatcherImpl<'a> where MatcherImpl: NewMatcher;

    fn into_matcher<'a>(self, s: &'a str) -> Self::MatcherImpl<'a>;
}

unsafe impl<'a, P> Pattern<'a> for P where P: NewPattern {
    type MatcherImpl = <Self as NewPattern>::MatcherImpl<'a>;

    fn into_matcher(self, haystack: &'a str) -> Self::MatcherImpl {
        <Self as NewPattern>::into_matcher(self, haystack)
    }
}

unsafe trait NewMatcher for Self<'_> {
    fn haystack<'a>(self: &Self<'a>) -> &'a str;
    fn next_match<'a>(self: &mut Self<'a>) -> Option<(uint, uint)>;
}

unsafe impl<'a, M> Matcher<'a> for M<'a> where M: NewMatcher {
    fn haystack(&self) -> &'a str {
        <M as NewMatcher>::haystack(self)
    }
    fn next_match(&mut self) -> Option<(uint, uint)> {
        <M as NewMatcher>::next_match(self)
    }
}
```

Based on coherency experiments and assumptions about how future HKT will work,
the author is assuming that the above implementation will work, but can not experimentally prove it.

In order for these new traits to fully replace the old ones without getting in their way,
the old ones need to not be defined in a way that makes them "final".
That is, they should be defined in their own submodule, like `str::pattern` that can grow
a sister module like `str::newpattern`, and not be exported in a global place like `str` or even
the `prelude` (which would be unneeded anyway).

# Drawbacks

- It complicates the whole machinery and API behind the implementation of matching on string patterns.
- The no-HKT-lifetime-workaround wart might be to confusing for something as commonplace as the string API.
- This add a few layers of generics, so compilation times and micro optimizations might suffer.

# Alternatives

## Alternatives in general

- Keep status quo, with all issues listed at the beginning.
- Stabilize on hardcoded variants, eg providing both `contains` and `contains_str`.
  Similar to status quo, but no `CharEq` and thus no generics.

## Primary alternatives in details of this proposal

The author identified two alternatives that might still give the same desired API long-term.
The biggest wart is the lifetime parameter on the two trait families, so both try to avoid it:

- Stabilize on a variant around `CharEq` - This would mean hardcoded `_str` methods,
  generic `CharEq` methods, and no extensibility to types like `Regex`, but has a
  upgrade path for later upgrading `CharEq` to a full-fledged, HKT-using `Pattern` API, by providing
  back-comp generic impls.
- Remove the lifetimes on `Matcher` and `Pattern` by requiring users of the API to store the haystack slice
  themselves, duplicating it in the in-memory representation.

## Other alternatives in details of this proposal

- Remove the lifetime parameter on `Pattern` and `Matcher` by making them fully unsafe API's,
  and require implementations to unsafely transmuting away and back the lifetime of the haystack slice.
- Remove `unsafe` from the API by not marking the `Matcher` traits as `unsafe`, requiring users of the API
  to explicitly check every match on validity in regard to utf8 boundaries.
- Allow to opt-in the `unsafe` traits by providing parallel safe and unsafe `Matcher` traits or methods,
  with the one per default implemented in terms of the other.
- Turn `Pattern` into `Pattern` and `ReversePattern`, starting the forward-reverse split at the level of
  patterns directly. The two would still be in a inherits-from relationship like
  `Matcher` and `ReverseMatcher`, and be interchangeable if the later also implement `DoubleEndedMatcher`,
  but on the `str` API `where` clauses like `where P: Pattern<'a>, P::MatcherImpl: ReverseMatcher<'a>`
  would turn into `where P: ReversePattern<'a>`.

# Unresolved questions

- Concrete performance is untested compared to the current situation.
- Should the API split in regard to forward-reverse matching be as symmetrical as possible,
  or as minimal as possible?
  In the first case, iterators like `Matches` and `RMatches` could both implement `DoubleEndedIterator` if a
  `DoubleEndedMatcher` exists, in the latter only `Matches` would, with `RMatches` only providing the
  minimum to support reverse operation.

# Additional extensions

A similar abstraction system could be implemented for `String` APIs, so that for example `string.push("foo")`,
`string.push('f')`, `string.push('f'.to_ascii())` all work by using something like a `StringSource` trait.

This would allow operations like `s.replace(&regex!(...), "foo")`,
which would be a method generic over both the pattern matched and the string fragment it gets replaced with:

```rust
fn replace<P, S>(&mut self, pat: P, with: S) where P: Pattern, S: StringSource { /* ... */ }
```
