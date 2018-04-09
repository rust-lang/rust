- Feature Name: `os_str_pattern`
- Start Date: 2018-01-16
- RFC PR: [rust-lang/rfcs#2295](https://github.com/rust-lang/rfcs/pull/2295)
- Rust Issue: [rust-lang/rust#49802](https://github.com/rust-lang/rust/issues/49802)

# Summary
[summary]: #summary

Generalize the WTF-8 encoding to allow `OsStr` to use the pattern API methods.

# Motivation
[motivation]: #motivation

`OsStr` is missing many common string methods compared to the standard `str` or even `[u8]`. There
have been numerous attempts to expand the API surface, the latest one being [RFC #1309], which
leads to an attempt to [revamp the `std::pattern::Pattern` API][Kimundi/rust_pattern_api_v2], but
eventually closed due to inactivity and lack of resource.

Over the past several years, there has been numerous requests and attempts to implement these
missing functions in particular `OsStr::starts_with` ([1][#22741], [2][#26499], [3][#40300],
[4][urlo #10403], [5][irlo #6277], [6][os-str-generic]).

The main difficulty applying `str` APIs to `OsStr` is [WTF-8]. A surrogate pair (e.g. U+10000 =
`d800 dc00`) is encoded as a 4-byte sequence (`f0 90 80 80`) similar to UTF-8, but an unpaired
surrogate (e.g. U+D800 alone) is encoded as a completely distinct 3-byte sequence (`ed a0 80`).
Naively extending the slice-based pattern API will not work, e.g. you cannot find any `ed a0 80`
inside `f0 90 80 80`, so `.starts_with()` is going to be more complex, and `.split()` certainly
cannot borrow a well-formed WTF-8 slice from it.

The solution proposed by RFC #1309 is to create two sets of APIs. One, `.contains_os()`,
`.starts_with_os()`, `.ends_with_os()` and `.replace()` which do not require borrowing, will support
using `&OsStr` as input. The rest like `.split()`, `.matches()` and `.trim()` which require
borrowing, will only accept UTF-8 strings as input.

The “pattern 2.0” API does not split into two sets of APIs, but will panic when the search string
starts with or ends with an unpaired surrogate.

We feel that these designs are not elegant enough. This RFC attempts to fix the problem by going one
level lower, by generalizing WTF-8 so that splitting a surrogate pair is allowed, so we could search
an `OsStr` with an `OsStr` using a single Pattern API without panicking.

[Kimundi/rust_pattern_api_v2]: https://github.com/Kimundi/rust_pattern_api_v2
[RFC #1309]: https://github.com/rust-lang/rfcs/pull/1309
[#22741]: https://github.com/rust-lang/rust/issues/22741
[#26499]: https://github.com/rust-lang/rust/issues/26499
[#40300]: https://github.com/rust-lang/rust/issues/40300
[urlo #10403]: https://users.rust-lang.org/t/comparing-osstr-for-prefixes-and-suffixes/10403
[irlo #6277]: https://internals.rust-lang.org/t/make-std-os-unix-ffi-osstrext-cross-platform/6277
[os-str-generic]: https://docs.rs/os-str-generic
[WTF-8]: https://simonsapin.github.io/wtf-8/

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

The following new methods are now available to `OsStr`. They behave the same as their counterpart in
`str`.

```rust
impl OsStr {
    pub fn contains<'a, P>(&'a self, pat: P) -> bool
    where
        P: Pattern<&'a Self>;

    pub fn starts_with<'a, P>(&'a self, pat: P) -> bool
    where
        P: Pattern<&'a Self>;

    pub fn ends_with<'a, P>(&'a self, pat: P) -> bool
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    pub fn find<'a, P>(&'a self, pat: P) -> Option<usize>
    where
        P: Pattern<&'a Self>;

    pub fn rfind<'a, P>(&'a self, pat: P) -> Option<usize>
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    /// Finds the first range of this string which contains the pattern.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let path = OsStr::new("/usr/bin/bash");
    /// let range = path.find_range("/b");
    /// assert_eq!(range, Some(4..6));
    /// assert_eq!(path[range.unwrap()], OsStr::new("/b"));
    /// ```
    pub fn find_range<'a, P>(&'a self, pat: P) -> Option<Range<usize>>
    where
        P: Pattern<&'a Self>;

    /// Finds the last range of this string which contains the pattern.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let path = OsStr::new("/usr/bin/bash");
    /// let range = path.rfind_range("/b");
    /// assert_eq!(range, Some(8..10));
    /// assert_eq!(path[range.unwrap()], OsStr::new("/b"));
    /// ```
    pub fn rfind_range<'a, P>(&'a self, pat: P) -> Option<Range<usize>>
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    // (Note: these should return a concrete iterator type instead of `impl Trait`.
    //  For ease of explanation the concrete type is not listed here.)
    pub fn split<'a, P>(&'a self, pat: P) -> impl Iterator<Item = &'a Self>
    where
        P: Pattern<&'a Self>;

    pub fn rsplit<'a, P>(&'a self, pat: P) -> impl Iterator<Item = &'a Self>
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    pub fn split_terminator<'a, P>(&'a self, pat: P) -> impl Iterator<Item = &'a Self>
    where
        P: Pattern<&'a Self>;

    pub fn rsplit_terminator<'a, P>(&'a self, pat: P) -> impl Iterator<Item = &'a Self>
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    pub fn splitn<'a, P>(&'a self, n: usize, pat: P) -> impl Iterator<Item = &'a Self>
    where
        P: Pattern<&'a Self>;

    pub fn rsplitn<'a, P>(&'a self, n: usize, pat: P) -> impl Iterator<Item = &'a Self>
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    pub fn matches<'a, P>(&'a self, pat: P) -> impl Iterator<Item = &'a Self>
    where
        P: Pattern<&'a Self>;

    pub fn rmatches<'a, P>(&self, pat: P) -> impl Iterator<Item = &'a Self>
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    pub fn match_indices<'a, P>(&self, pat: P) -> impl Iterator<Item = (usize, &'a Self)>
    where
        P: Pattern<&'a Self>;

    pub fn rmatch_indices<'a, P>(&self, pat: P) -> impl Iterator<Item = (usize, &'a Self)>
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    // this is new
    pub fn match_ranges<'a, P>(&'a self, pat: P) -> impl Iterator<Item = (Range<usize>, &'a Self)>
    where
        P: Pattern<&'a Self>;

    // this is new
    pub fn rmatch_ranges<'a, P>(&'a self, pat: P) -> impl Iterator<Item = (Range<usize>, &'a Self)>
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    pub fn trim_matches<'a, P>(&'a self, pat: P) -> &'a Self
    where
        P: Pattern<&'a Self>,
        P::Searcher: DoubleEndedSearcher<&'a Self>;

    pub fn trim_left_matches<'a, P>(&'a self, pat: P) -> &'a Self
    where
        P: Pattern<&'a Self>;

    pub fn trim_right_matches<'a, P>(&'a self, pat: P) -> &'a Self
    where
        P: Pattern<&'a Self>,
        P::Searcher: ReverseSearcher<&'a Self>;

    pub fn replace<'a, P>(&'a self, from: P, to: &'a Self) -> Self::Owned
    where
        P: Pattern<&'a Self>;

    pub fn replacen<'a, P>(&'a self, from: P, to: &'a Self, count: usize) -> Self::Owned
    where
        P: Pattern<&'a Self>;
}
```

We also allow slicing an `OsStr`.

```rust
impl Index<RangeFull> for OsStr { ... }
impl Index<RangeFrom<usize>> for OsStr { ... }
impl Index<RangeTo<usize>> for OsStr { ... }
impl Index<Range<usize>> for OsStr { ... }
```

Example:

```rust
// (assume we are on Windows)

let path = OsStr::new(r"C:\Users\Admin\😀\😁😂😃😄.txt");
// can use starts_with, ends_with
assert!(path.starts_with(OsStr::new(r"C:\")));
assert!(path.ends_with(OsStr::new(".txt"));
// can use rfind_range to get the range of substring
let last_backslash = path.rfind_range(OsStr::new(r"\")).unwrap();
assert_eq!(last_backslash, 16..17);
// can perform slicing.
let file_name = &path[last_backslash.end..];
// can perform splitting, even if it results in invalid Unicode!
let mut parts = file_name.split(&*OsString::from_wide(&[0xd83d]));
assert_eq!(parts.next(), Some(OsStr::new("")));
assert_eq!(parts.next(), Some(&*OsString::from_wide(&[0xde01])));
assert_eq!(parts.next(), Some(&*OsString::from_wide(&[0xde02])));
assert_eq!(parts.next(), Some(&*OsString::from_wide(&[0xde03])));
assert_eq!(parts.next(), Some(&*OsString::from_wide(&[0xde04, 0x2e, 0x74, 0x78, 0x74])));
assert_eq!(parts.next(), None);
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

It is trivial to apply the pattern API to `OsStr` on platforms where it is just an `[u8]`. The main
difficulty is on Windows where it is an `[u16]` encoded as WTF-8. This RFC thus focuses on Windows.

We will generalize the encoding of `OsStr` to “[OMG-WTF-8]” which specifies these two capabilities:

1. Slicing a surrogate pair in half:

    ```rust
    let s = OsStr::new("\u{10000}");
    assert_eq!(&s[..2], &*OsString::from_wide(&[0xd800]));
    assert_eq!(&s[2..], &*OsString::from_wide(&[0xdc00]));
    ```

2. Finding a surrogate code point, no matter paired or unpaired:

    ```rust
    let needle = OsString::from_wide(&[0xdc00]);
    assert_eq!(OsStr::new("\u{10000}").find(&needle), Some(2));
    assert_eq!(OsString::from_wide(&[0x3f, 0xdc00]).find(&needle), Some(1));
    ```

These allow us to implement the “Pattern 1.5” API for all `OsStr` without panicking. Implementation
detail can be found in the [`omgwtf8` package][OMG-WTF-8].

[OMG-WTF-8]: https://github.com/kennytm/omgwtf8

## Slicing

A surrogate pair is a 4-byte sequence in both UTF-8 and WTF-8. We support slicing it in half by
representing the high surrogate by the first 3 bytes, and the low surrogate by the last 3 bytes.

```
"\u{10000}"      = f0 90 80 80
"\u{10000}"[..2] = f0 90 80
"\u{10000}"[2..] =    90 80 80
```

The index splitting the surrogate pair will be positioned at the middle of the 4-byte sequence
(index "2" in the above example).

Note that this means:

1. `x[..i]` and `x[i..]` will have overlapping parts. This makes `OsStr::split_at_mut` (if exists)
    unable to split a surrogate pair in half. This also means `Pattern<&mut OsStr>` cannot be
    implemented for `&OsStr`.
2. The length of `x[..n]` may be longer than `n`.

### Platform-agnostic guarantees

If an index points to an invalid position (e.g. `\u{1000}[1..]` or `"\u{10000}"[1..]` or
`"\u{10000}"[3..]`), a panic will be raised, similar to that of `str`. The following are guaranteed
to be valid positions on all platforms:

* `0`.
* `self.len()`.
* The returned indices from `find()`, `rfind()`, `match_indices()` and `rmatch_indices()`.
* The returned ranges from `find_range()`, `rfind_range()`, `match_ranges()` and `rmatch_ranges()`.

Index arithmetic is wrong for `OsStr`, i.e. `i + n` may not produce the correct index (see
[Drawbacks](#drawbacks)).

For WTF-8 encoding on Windows, we define:

* boundary of a character or surrogate byte sequence is Valid.
* middle (byte 2) of a 4-byte sequence is Valid.
* interior of a 2- or 3-byte sequence is Invalid.
* byte 1 or 3 of a 4-byte sequence is Invalid.

Outside of Windows where the `OsStr` consists of arbitrary bytes, all numbers within
`0 ..= self.len()` are considered a valid index. This is because we want to allow
`os_str.find(OsStr::from_bytes(b"\xff"))`, and thus cannot use UTF-8 to reason with a Unix `OsStr`.

Note that we have never guaranteed the actual `OsStr` encoding, these should only be considered an
implementation detail.

## Comparison and storage

All `OsStr` strings with sliced 4-byte sequence can be converted back to proper WTF-8 with an O(1)
transformation:

* If the string starts with `[\x80-\xbf]{3}`, replace these 3 bytes with the canonical low surrogate
    encoding.
* If the string ends with `[\xf0-\xf4][\x80-\xbf]{2}`, replace these 3 bytes with the canonical high
    surrogate encoding.

We can this transformation “*canonicalization*”.

All owned `OsStr` should be canonicalized to contain well-formed WTF-8 only: `Box<OsStr>`,
`Rc<OsStr>`, `Arc<OsStr>` and `OsString`.

Two `OsStr` are compared equal if they have the same canonicalization. This may slightly reduce the
performance with a constant overhead, since there would be more checking involving the first and
last three bytes.

## Matching

When an `OsStr` is used for matching, an unpaired low surrogate at the beginning and unpaired high
surrogate at the end must be replaced by regular expressions that match all pre-canonicalization
possibilities. For instance, matching for `xxxx\u{d9ab}` would create the following regex:

```
xxxx(
    \xed\xa6\xab        # canonical representation
|
    \xf2\x86[\xb0-\xbf] # split representation
)
```

and matching for `\u{dcef}xxxx` with create the following regex:

```
(
    \xed\xb3\xaf                        # canonical representation
|
    [\x80-\xbf][\x83\x93\xa3\xb3]\xaf   # split representation
)xxxx
```

After finding a match, if the end points to the middle of a 4-byte sequence, the search engine
should move backward by 2 bytes before continuing. This ensure searching for `\u{dc00}\u{d800}` in
`\u{10000}\u{10000}\u{10000}` will properly yield 2 matches.

## Pattern API

As of Rust 1.25, we can search a `&str` using a character, a character set or another string,
powered by [RFC #528](https://github.com/rust-lang/rfcs/pull/528) a.k.a. “Pattern API 1.0”.

There are some drafts to generalize this so that we could retain mutability and search in more types
such as `&[T]` and `&OsStr`, as described in various comments
(“[v1.5](https://github.com/rust-lang/rust/issues/27721#issuecomment-185405392)” and
“[v2.0](https://github.com/rust-lang/rfcs/pull/1309#issuecomment-214030263)”). A proper RFC has not
been proposed so far.

This RFC assumes the target of generalizing the Pattern API beyond `&str` is accepted, enabling us
to provide a uniform search API between different types of haystack and needles. However, this RFC
does not rely on a generalized Pattern API. If this RFC is stabilized without a generalized Pattern
API, the new methods described in the [Guide-level explanation][guide-level-explanation] section can
take `&OsStr` instead of `impl Pattern<&OsStr>`, but this may hurt future compatibility due to
inference breakage if generalized Pattern API is indeed implemented.

Assuming we do want to generalize Pattern API, the implementor should note the issue of splitting a
surrogate pair:

1. A match which starts with a low surrogate will point to byte 1 of the 4-byte sequence
2. An index always point to byte 2 of the 4-byte sequence
3. A match which ends with a high surrogate will point to byte 3 of the 4-byte sequence

Implementation should note these different offsets when converting between different kinds of
cursors. In the [`omgwtf8::pattern` module](https://docs.rs/omgwtf8/*/omgwtf8/pattern/index.html),
based on the “v1.5” draft, this behavior is enforced in the API design by using distinct types for
the start and end cursors.

The following outlines the generalized Pattern API which could work for `&OsStr`:

```rust
// in module `core::pattern`:

pub trait Pattern<H: Haystack>: Sized {
    type Searcher: Searcher<H>;
    fn into_searcher(self, haystack: H) -> Self::Searcher;
    fn is_contained_in(self, haystack: H) -> bool;
    fn is_prefix_of(self, haystack: H) -> bool;
    fn is_suffix_of(self, haystack: H) -> bool where Self::Searcher: ReverseSearcher<H>;
}

pub trait Searcher<H: Haystack> {
    fn haystack(&self) -> H;
    fn next_match(&mut self) -> Option<(H::StartCursor, H::EndCursor)>;
    fn next_reject(&mut self) -> Option<(H::StartCursor, H::EndCursor)>;
}

pub trait ReverseSearcher<H: Haystack>: Searcher<H> {
    fn next_match_back(&mut self) -> Option<(H::StartCursor, H::EndCursor)>;
    fn next_reject_back(&mut self) -> Option<(H::StartCursor, H::EndCursor)>;
}

pub trait DoubleEndedSearcher<H: Haystack>: ReverseSearcher<H> {}

// equivalent to SearchPtrs in "Pattern API 1.5"
// and PatternHaystack in "Pattern API 2.0"
pub trait Haystack: Sized {
    type StartCursor: Copy + PartialOrd<Self::EndCursor>;
    type EndCursor: Copy + PartialOrd<Self::StartCursor>;

    // The following 5 methods are same as those in "Pattern API 1.5"
    // except the cursor type is split into two.
    fn cursor_at_front(hs: &Self) -> Self::StartCursor;
    fn cursor_at_back(hs: &Self) -> Self::EndCursor;
    unsafe fn start_cursor_to_offset(hs: &Self, cur: Self::StartCursor) -> usize;
    unsafe fn end_cursor_to_offset(hs: &Self, cur: Self::EndCursor) -> usize;
    unsafe fn range_to_self(hs: Self, start: Self::StartCursor, end: Self::EndCursor) -> Self;

    // And then we want to swap between the two cursor types
    unsafe fn start_to_end_cursor(hs: &Self, cur: Self::StartCursor) -> Self::EndCursor;
    unsafe fn end_to_start_cursor(hs: &Self, cur: Self::EndCursor) -> Self::StartCursor;
}
```

For the `&OsStr` haystack, we define both `StartCursor` and `EndCursor` as `*const u8`.

The `start_to_end_cursor` function will return `cur + 2` if we find that `cur` points to the middle
of a 4-byte sequence.

The `start_cursor_to_offset` function will return `cur - hs + 1` if we find that `cur` points to the
middle of a 4-byte sequenced.

These type safety measures ensure functions utilizing a generic `Pattern` can get the correctly
overlapping slices when splitting a surrogate pair.

```rust
// (actual code implementing `.split()`)
match self.matcher.next_match() {
    Some((a, b)) => unsafe {
        let haystack = self.matcher.haystack();
        let a = H::start_to_end_cursor(&haystack, a);
        let b = H::end_to_start_cursor(&haystack, b);
        let elt = H::range_to_self(haystack, self.start, a);
        // ^ without `start_to_end_cursor`, the slice `elt` may be short by 2 bytes
        self.start = b;
        // ^ without `end_to_start_cursor`, the next starting position may skip 2 bytes
        Some(elt)
    },
    None => self.get_end(),
}
```

# Drawbacks
[drawbacks]: #drawbacks

* **It breaks the invariant `x[..n].len() == n`.**

    Note that `OsStr` did not provide a slicing operator, and it already violated the invariant
    `(x + y).len() == x.len() + y.len()`.

* **A surrogate code point may be 2 or 3 indices long depending on context.**

    This means code using `x[i..(i+n)]` may give wrong result.

    ```rust
    let needle = OsString::from_wide(&[0xdc00]);
    let haystack = OsStr::new("\u{10000}a");
    let index = haystack.find(&needle).unwrap();
    let matched = &haystack[index..(index + needle.len())];
    // `matched` will contain "\u{dc00}a" instead of "\u{dc00}".
    ```

    As a workaround, we introduced `find_range` and `match_ranges`. Note that this is already a
    problem to solve if we want to make `Regex` a pattern of strings.

# Rationale and alternatives
[alternatives]: #alternatives

## Indivisible surrogate pair

This RFC is the only design which allows borrowing a sub-slice of a surrogate code point from a
surrogate pair.

An alternative is keep using the vanilla WTF-8, and treat a surrogate pair as an atomic entity:
makes it impossible to split a surrogate pair after it is formed. The advantages are that

* The pattern API becomes a simple substring search.
* Slicing behavior is consistent with `str`.

There are two potential implementations when we want to match with an unpaired surrogate:

1. **Declare that a surrogate pair does not contain the unpaired surrogate**, i.e. make
    `"\u{10000}".find("\u{d800}")` return `None`. An unpaired surrogate can only be used to match
    another unpaired surrogate.

    If we choose this, it means `x.find(z).is_some()` does not imply `(x + y).find(z).is_some()`.

2. **Disallow matching when the pattern contains an unpaired surrogate at the boundary**, i.e. make
    `"\u{10000}".find("\u{d800}")` panic. This is the approach chosen by “Pattern API 2.0”.

Note that, for consistency, we need to make `"\u{10000}".starts_with("\u{d800}")` return `false` or
panic.

## Slicing at real byte offset

The current RFC defines the index that splits a surrogate pair into half at byte 2 of the 4-byte
sequence. This has the drawback of `"\u{10000}"[..2].len() == 3`, and caused index arithmetic to be
wrong.

```
"\u{10000}"      = f0 90 80 80
"\u{10000}"[..2] = f0 90 80
"\u{10000}"[2..] =    90 80 80
```

The main advantage of this scheme is we could use the same number as the start and end index.

```rust
let s = OsStr::new("\u{10000}");
assert_eq!(s.len(), 4);
let index = s.find('\u{dc00}').unwrap();
let right = &s[index..];  // [90 80 80]
let left = &s[..index];   // [f0 90 80]
```

An alternative make the index refer to the real byte offsets:

```
"\u{10000}"      = f0 90 80 80
"\u{10000}"[..3] = f0 90 80
"\u{10000}"[1..] =    90 80 80
```

However the question would be, what should `s[..1]` do?

* **Panic** — But this means we cannot get `left`. We could inspect the raw bytes of `s` itself and
    perform `&s[..(index + 2)]`, but we never explicitly exposed the encoding of `OsStr`, so we
    cannot read a single byte and thus impossible to do this.

* **Treat as same as `s[..3]`** — But then this inherits all the disadvantages of using 2 as valid
    index, plus we need to consider whether `s[1..3]` and `s[3..1]` should be valid.

Given these, we decided not to treat the real byte offsets as valid indices.

# Unresolved questions
[unresolved]: #unresolved-questions

None yet.
