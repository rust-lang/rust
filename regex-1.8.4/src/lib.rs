/*!
This crate provides a library for parsing, compiling, and executing regular
expressions. Its syntax is similar to Perl-style regular expressions, but lacks
a few features like look around and backreferences. In exchange, all searches
execute in linear time with respect to the size of the regular expression and
search text.

This crate's documentation provides some simple examples, describes
[Unicode support](#unicode) and exhaustively lists the
[supported syntax](#syntax).

For more specific details on the API for regular expressions, please see the
documentation for the [`Regex`](struct.Regex.html) type.

# Usage

This crate is [on crates.io](https://crates.io/crates/regex) and can be
used by adding `regex` to your dependencies in your project's `Cargo.toml`.

```toml
[dependencies]
regex = "1"
```

# Example: find a date

General use of regular expressions in this package involves compiling an
expression and then using it to search, split or replace text. For example,
to confirm that some text resembles a date:

```rust
use regex::Regex;
let re = Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
assert!(re.is_match("2014-01-01"));
```

Notice the use of the `^` and `$` anchors. In this crate, every expression
is executed with an implicit `.*?` at the beginning and end, which allows
it to match anywhere in the text. Anchors can be used to ensure that the
full text matches an expression.

This example also demonstrates the utility of
[raw strings](https://doc.rust-lang.org/stable/reference/tokens.html#raw-string-literals)
in Rust, which
are just like regular strings except they are prefixed with an `r` and do
not process any escape sequences. For example, `"\\d"` is the same
expression as `r"\d"`.

# Example: Avoid compiling the same regex in a loop

It is an anti-pattern to compile the same regular expression in a loop
since compilation is typically expensive. (It takes anywhere from a few
microseconds to a few **milliseconds** depending on the size of the
regex.) Not only is compilation itself expensive, but this also prevents
optimizations that reuse allocations internally to the matching engines.

In Rust, it can sometimes be a pain to pass regular expressions around if
they're used from inside a helper function. Instead, we recommend using the
[`lazy_static`](https://crates.io/crates/lazy_static) crate to ensure that
regular expressions are compiled exactly once.

For example:

```rust
use lazy_static::lazy_static;
use regex::Regex;

fn some_helper_function(text: &str) -> bool {
    lazy_static! {
        static ref RE: Regex = Regex::new("...").unwrap();
    }
    RE.is_match(text)
}

fn main() {}
```

Specifically, in this example, the regex will be compiled when it is used for
the first time. On subsequent uses, it will reuse the previous compilation.

# Example: iterating over capture groups

This crate provides convenient iterators for matching an expression
repeatedly against a search string to find successive non-overlapping
matches. For example, to find all dates in a string and be able to access
them by their component pieces:

```rust
# use regex::Regex;
# fn main() {
let re = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();
let text = "2012-03-14, 2013-01-01 and 2014-07-05";
for cap in re.captures_iter(text) {
    println!("Month: {} Day: {} Year: {}", &cap[2], &cap[3], &cap[1]);
}
// Output:
// Month: 03 Day: 14 Year: 2012
// Month: 01 Day: 01 Year: 2013
// Month: 07 Day: 05 Year: 2014
# }
```

Notice that the year is in the capture group indexed at `1`. This is
because the *entire match* is stored in the capture group at index `0`.

# Example: replacement with named capture groups

Building on the previous example, perhaps we'd like to rearrange the date
formats. This can be done with text replacement. But to make the code
clearer, we can *name*  our capture groups and use those names as variables
in our replacement text:

```rust
# use regex::Regex;
# fn main() {
let re = Regex::new(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})").unwrap();
let before = "2012-03-14, 2013-01-01 and 2014-07-05";
let after = re.replace_all(before, "$m/$d/$y");
assert_eq!(after, "03/14/2012, 01/01/2013 and 07/05/2014");
# }
```

The `replace` methods are actually polymorphic in the replacement, which
provides more flexibility than is seen here. (See the documentation for
`Regex::replace` for more details.)

Note that if your regex gets complicated, you can use the `x` flag to
enable insignificant whitespace mode, which also lets you write comments:

```rust
# use regex::Regex;
# fn main() {
let re = Regex::new(r"(?x)
  (?P<y>\d{4}) # the year
  -
  (?P<m>\d{2}) # the month
  -
  (?P<d>\d{2}) # the day
").unwrap();
let before = "2012-03-14, 2013-01-01 and 2014-07-05";
let after = re.replace_all(before, "$m/$d/$y");
assert_eq!(after, "03/14/2012, 01/01/2013 and 07/05/2014");
# }
```

If you wish to match against whitespace in this mode, you can still use `\s`,
`\n`, `\t`, etc. For escaping a single space character, you can escape it
directly with `\ `, use its hex character code `\x20` or temporarily disable
the `x` flag, e.g., `(?-x: )`.

# Example: match multiple regular expressions simultaneously

This demonstrates how to use a `RegexSet` to match multiple (possibly
overlapping) regular expressions in a single scan of the search text:

```rust
use regex::RegexSet;

let set = RegexSet::new(&[
    r"\w+",
    r"\d+",
    r"\pL+",
    r"foo",
    r"bar",
    r"barfoo",
    r"foobar",
]).unwrap();

// Iterate over and collect all of the matches.
let matches: Vec<_> = set.matches("foobar").into_iter().collect();
assert_eq!(matches, vec![0, 2, 3, 4, 6]);

// You can also test whether a particular regex matched:
let matches = set.matches("foobar");
assert!(!matches.matched(5));
assert!(matches.matched(6));
```

# Pay for what you use

With respect to searching text with a regular expression, there are three
questions that can be asked:

1. Does the text match this expression?
2. If so, where does it match?
3. Where did the capturing groups match?

Generally speaking, this crate could provide a function to answer only #3,
which would subsume #1 and #2 automatically. However, it can be significantly
more expensive to compute the location of capturing group matches, so it's best
not to do it if you don't need to.

Therefore, only use what you need. For example, don't use `find` if you
only need to test if an expression matches a string. (Use `is_match`
instead.)

# Unicode

This implementation executes regular expressions **only** on valid UTF-8
while exposing match locations as byte indices into the search string. (To
relax this restriction, use the [`bytes`](bytes/index.html) sub-module.)
Conceptually, the regex engine works by matching a haystack as if it were a
sequence of Unicode scalar values.

Only simple case folding is supported. Namely, when matching
case-insensitively, the characters are first mapped using the "simple" case
folding rules defined by Unicode.

Regular expressions themselves are **only** interpreted as a sequence of
Unicode scalar values. This means you can use Unicode characters directly
in your expression:

```rust
# use regex::Regex;
# fn main() {
let re = Regex::new(r"(?i)Δ+").unwrap();
let mat = re.find("ΔδΔ").unwrap();
assert_eq!((mat.start(), mat.end()), (0, 6));
# }
```

Most features of the regular expressions in this crate are Unicode aware. Here
are some examples:

* `.` will match any valid UTF-8 encoded Unicode scalar value except for `\n`.
  (To also match `\n`, enable the `s` flag, e.g., `(?s:.)`.)
* `\w`, `\d` and `\s` are Unicode aware. For example, `\s` will match all forms
  of whitespace categorized by Unicode.
* `\b` matches a Unicode word boundary.
* Negated character classes like `[^a]` match all Unicode scalar values except
  for `a`.
* `^` and `$` are **not** Unicode aware in multi-line mode. Namely, they only
  recognize `\n` and not any of the other forms of line terminators defined
  by Unicode.

Unicode general categories, scripts, script extensions, ages and a smattering
of boolean properties are available as character classes. For example, you can
match a sequence of numerals, Greek or Cherokee letters:

```rust
# use regex::Regex;
# fn main() {
let re = Regex::new(r"[\pN\p{Greek}\p{Cherokee}]+").unwrap();
let mat = re.find("abcΔᎠβⅠᏴγδⅡxyz").unwrap();
assert_eq!((mat.start(), mat.end()), (3, 23));
# }
```

For a more detailed breakdown of Unicode support with respect to
[UTS#18](https://unicode.org/reports/tr18/),
please see the
[UNICODE](https://github.com/rust-lang/regex/blob/master/UNICODE.md)
document in the root of the regex repository.

# Opt out of Unicode support

The `bytes` sub-module provides a `Regex` type that can be used to match
on `&[u8]`. By default, text is interpreted as UTF-8 just like it is with
the main `Regex` type. However, this behavior can be disabled by turning
off the `u` flag, even if doing so could result in matching invalid UTF-8.
For example, when the `u` flag is disabled, `.` will match any byte instead
of any Unicode scalar value.

Disabling the `u` flag is also possible with the standard `&str`-based `Regex`
type, but it is only allowed where the UTF-8 invariant is maintained. For
example, `(?-u:\w)` is an ASCII-only `\w` character class and is legal in an
`&str`-based `Regex`, but `(?-u:\xFF)` will attempt to match the raw byte
`\xFF`, which is invalid UTF-8 and therefore is illegal in `&str`-based
regexes.

Finally, since Unicode support requires bundling large Unicode data
tables, this crate exposes knobs to disable the compilation of those
data tables, which can be useful for shrinking binary size and reducing
compilation times. For details on how to do that, see the section on [crate
features](#crate-features).

# Syntax

The syntax supported in this crate is documented below.

Note that the regular expression parser and abstract syntax are exposed in
a separate crate, [`regex-syntax`](https://docs.rs/regex-syntax).

## Matching one character

<pre class="rust">
.             any character except new line (includes new line with s flag)
\d            digit (\p{Nd})
\D            not digit
\pX           Unicode character class identified by a one-letter name
\p{Greek}     Unicode character class (general category or script)
\PX           Negated Unicode character class identified by a one-letter name
\P{Greek}     negated Unicode character class (general category or script)
</pre>

### Character classes

<pre class="rust">
[xyz]         A character class matching either x, y or z (union).
[^xyz]        A character class matching any character except x, y and z.
[a-z]         A character class matching any character in range a-z.
[[:alpha:]]   ASCII character class ([A-Za-z])
[[:^alpha:]]  Negated ASCII character class ([^A-Za-z])
[x[^xyz]]     Nested/grouping character class (matching any character except y and z)
[a-y&&xyz]    Intersection (matching x or y)
[0-9&&[^4]]   Subtraction using intersection and negation (matching 0-9 except 4)
[0-9--4]      Direct subtraction (matching 0-9 except 4)
[a-g~~b-h]    Symmetric difference (matching `a` and `h` only)
[\[\]]        Escaping in character classes (matching [ or ])
</pre>

Any named character class may appear inside a bracketed `[...]` character
class. For example, `[\p{Greek}[:digit:]]` matches any Greek or ASCII
digit. `[\p{Greek}&&\pL]` matches Greek letters.

Precedence in character classes, from most binding to least:

1. Ranges: `a-cd` == `[a-c]d`
2. Union: `ab&&bc` == `[ab]&&[bc]`
3. Intersection: `^a-z&&b` == `^[a-z&&b]`
4. Negation

## Composites

<pre class="rust">
xy    concatenation (x followed by y)
x|y   alternation (x or y, prefer x)
</pre>

This example shows how an alternation works, and what it means to prefer a
branch in the alternation over subsequent branches.

```
use regex::Regex;

let haystack = "samwise";
// If 'samwise' comes first in our alternation, then it is
// preferred as a match, even if the regex engine could
// technically detect that 'sam' led to a match earlier.
let re = Regex::new(r"samwise|sam").unwrap();
assert_eq!("samwise", re.find(haystack).unwrap().as_str());
// But if 'sam' comes first, then it will match instead.
// In this case, it is impossible for 'samwise' to match
// because 'sam' is a prefix of it.
let re = Regex::new(r"sam|samwise").unwrap();
assert_eq!("sam", re.find(haystack).unwrap().as_str());
```

## Repetitions

<pre class="rust">
x*        zero or more of x (greedy)
x+        one or more of x (greedy)
x?        zero or one of x (greedy)
x*?       zero or more of x (ungreedy/lazy)
x+?       one or more of x (ungreedy/lazy)
x??       zero or one of x (ungreedy/lazy)
x{n,m}    at least n x and at most m x (greedy)
x{n,}     at least n x (greedy)
x{n}      exactly n x
x{n,m}?   at least n x and at most m x (ungreedy/lazy)
x{n,}?    at least n x (ungreedy/lazy)
x{n}?     exactly n x
</pre>

## Empty matches

<pre class="rust">
^     the beginning of text (or start-of-line with multi-line mode)
$     the end of text (or end-of-line with multi-line mode)
\A    only the beginning of text (even with multi-line mode enabled)
\z    only the end of text (even with multi-line mode enabled)
\b    a Unicode word boundary (\w on one side and \W, \A, or \z on other)
\B    not a Unicode word boundary
</pre>

The empty regex is valid and matches the empty string. For example, the empty
regex matches `abc` at positions `0`, `1`, `2` and `3`.

## Grouping and flags

<pre class="rust">
(exp)          numbered capture group (indexed by opening parenthesis)
(?P&lt;name&gt;exp)  named (also numbered) capture group (names must be alpha-numeric)
(?&lt;name&gt;exp)   named (also numbered) capture group (names must be alpha-numeric)
(?:exp)        non-capturing group
(?flags)       set flags within current group
(?flags:exp)   set flags for exp (non-capturing)
</pre>

Capture group names must be any sequence of alpha-numeric Unicode codepoints,
in addition to `.`, `_`, `[` and `]`. Names must start with either an `_` or
an alphabetic codepoint. Alphabetic codepoints correspond to the `Alphabetic`
Unicode property, while numeric codepoints correspond to the union of the
`Decimal_Number`, `Letter_Number` and `Other_Number` general categories.

Flags are each a single character. For example, `(?x)` sets the flag `x`
and `(?-x)` clears the flag `x`. Multiple flags can be set or cleared at
the same time: `(?xy)` sets both the `x` and `y` flags and `(?x-y)` sets
the `x` flag and clears the `y` flag.

All flags are by default disabled unless stated otherwise. They are:

<pre class="rust">
i     case-insensitive: letters match both upper and lower case
m     multi-line mode: ^ and $ match begin/end of line
s     allow . to match \n
U     swap the meaning of x* and x*?
u     Unicode support (enabled by default)
x     verbose mode, ignores whitespace and allow line comments (starting with `#`)
</pre>

Note that in verbose mode, whitespace is ignored everywhere, including within
character classes. To insert whitespace, use its escaped form or a hex literal.
For example, `\ ` or `\x20` for an ASCII space.

Flags can be toggled within a pattern. Here's an example that matches
case-insensitively for the first part but case-sensitively for the second part:

```rust
# use regex::Regex;
# fn main() {
let re = Regex::new(r"(?i)a+(?-i)b+").unwrap();
let cap = re.captures("AaAaAbbBBBb").unwrap();
assert_eq!(&cap[0], "AaAaAbb");
# }
```

Notice that the `a+` matches either `a` or `A`, but the `b+` only matches
`b`.

Multi-line mode means `^` and `$` no longer match just at the beginning/end of
the input, but at the beginning/end of lines:

```
# use regex::Regex;
let re = Regex::new(r"(?m)^line \d+").unwrap();
let m = re.find("line one\nline 2\n").unwrap();
assert_eq!(m.as_str(), "line 2");
```

Note that `^` matches after new lines, even at the end of input:

```
# use regex::Regex;
let re = Regex::new(r"(?m)^").unwrap();
let m = re.find_iter("test\n").last().unwrap();
assert_eq!((m.start(), m.end()), (5, 5));
```

Here is an example that uses an ASCII word boundary instead of a Unicode
word boundary:

```rust
# use regex::Regex;
# fn main() {
let re = Regex::new(r"(?-u:\b).+(?-u:\b)").unwrap();
let cap = re.captures("$$abc$$").unwrap();
assert_eq!(&cap[0], "abc");
# }
```

## Escape sequences

<pre class="rust">
\*          literal *, works for any punctuation character: \.+*?()|[]{}^$
\a          bell (\x07)
\f          form feed (\x0C)
\t          horizontal tab
\n          new line
\r          carriage return
\v          vertical tab (\x0B)
\123        octal character code (up to three digits) (when enabled)
\x7F        hex character code (exactly two digits)
\x{10FFFF}  any hex character code corresponding to a Unicode code point
\u007F      hex character code (exactly four digits)
\u{7F}      any hex character code corresponding to a Unicode code point
\U0000007F  hex character code (exactly eight digits)
\U{7F}      any hex character code corresponding to a Unicode code point
</pre>

## Perl character classes (Unicode friendly)

These classes are based on the definitions provided in
[UTS#18](https://www.unicode.org/reports/tr18/#Compatibility_Properties):

<pre class="rust">
\d     digit (\p{Nd})
\D     not digit
\s     whitespace (\p{White_Space})
\S     not whitespace
\w     word character (\p{Alphabetic} + \p{M} + \d + \p{Pc} + \p{Join_Control})
\W     not word character
</pre>

## ASCII character classes

<pre class="rust">
[[:alnum:]]    alphanumeric ([0-9A-Za-z])
[[:alpha:]]    alphabetic ([A-Za-z])
[[:ascii:]]    ASCII ([\x00-\x7F])
[[:blank:]]    blank ([\t ])
[[:cntrl:]]    control ([\x00-\x1F\x7F])
[[:digit:]]    digits ([0-9])
[[:graph:]]    graphical ([!-~])
[[:lower:]]    lower case ([a-z])
[[:print:]]    printable ([ -~])
[[:punct:]]    punctuation ([!-/:-@\[-`{-~])
[[:space:]]    whitespace ([\t\n\v\f\r ])
[[:upper:]]    upper case ([A-Z])
[[:word:]]     word characters ([0-9A-Za-z_])
[[:xdigit:]]   hex digit ([0-9A-Fa-f])
</pre>

# Crate features

By default, this crate tries pretty hard to make regex matching both as fast
as possible and as correct as it can be, within reason. This means that there
is a lot of code dedicated to performance, the handling of Unicode data and the
Unicode data itself. Overall, this leads to more dependencies, larger binaries
and longer compile times.  This trade off may not be appropriate in all cases,
and indeed, even when all Unicode and performance features are disabled, one
is still left with a perfectly serviceable regex engine that will work well
in many cases.

This crate exposes a number of features for controlling that trade off. Some
of these features are strictly performance oriented, such that disabling them
won't result in a loss of functionality, but may result in worse performance.
Other features, such as the ones controlling the presence or absence of Unicode
data, can result in a loss of functionality. For example, if one disables the
`unicode-case` feature (described below), then compiling the regex `(?i)a`
will fail since Unicode case insensitivity is enabled by default. Instead,
callers must use `(?i-u)a` instead to disable Unicode case folding. Stated
differently, enabling or disabling any of the features below can only add or
subtract from the total set of valid regular expressions. Enabling or disabling
a feature will never modify the match semantics of a regular expression.

All features below are enabled by default.

### Ecosystem features

* **std** -
  When enabled, this will cause `regex` to use the standard library. Currently,
  disabling this feature will always result in a compilation error. It is
  intended to add `alloc`-only support to regex in the future.

### Performance features

* **perf** -
  Enables all performance related features. This feature is enabled by default
  and will always cover all features that improve performance, even if more
  are added in the future.
* **perf-dfa** -
  Enables the use of a lazy DFA for matching. The lazy DFA is used to compile
  portions of a regex to a very fast DFA on an as-needed basis. This can
  result in substantial speedups, usually by an order of magnitude on large
  haystacks. The lazy DFA does not bring in any new dependencies, but it can
  make compile times longer.
* **perf-inline** -
  Enables the use of aggressive inlining inside match routines. This reduces
  the overhead of each match. The aggressive inlining, however, increases
  compile times and binary size.
* **perf-literal** -
  Enables the use of literal optimizations for speeding up matches. In some
  cases, literal optimizations can result in speedups of _several_ orders of
  magnitude. Disabling this drops the `aho-corasick` and `memchr` dependencies.
* **perf-cache** -
  This feature used to enable a faster internal cache at the cost of using
  additional dependencies, but this is no longer an option. A fast internal
  cache is now used unconditionally with no additional dependencies. This may
  change in the future.

### Unicode features

* **unicode** -
  Enables all Unicode features. This feature is enabled by default, and will
  always cover all Unicode features, even if more are added in the future.
* **unicode-age** -
  Provide the data for the
  [Unicode `Age` property](https://www.unicode.org/reports/tr44/tr44-24.html#Character_Age).
  This makes it possible to use classes like `\p{Age:6.0}` to refer to all
  codepoints first introduced in Unicode 6.0
* **unicode-bool** -
  Provide the data for numerous Unicode boolean properties. The full list
  is not included here, but contains properties like `Alphabetic`, `Emoji`,
  `Lowercase`, `Math`, `Uppercase` and `White_Space`.
* **unicode-case** -
  Provide the data for case insensitive matching using
  [Unicode's "simple loose matches" specification](https://www.unicode.org/reports/tr18/#Simple_Loose_Matches).
* **unicode-gencat** -
  Provide the data for
  [Unicode general categories](https://www.unicode.org/reports/tr44/tr44-24.html#General_Category_Values).
  This includes, but is not limited to, `Decimal_Number`, `Letter`,
  `Math_Symbol`, `Number` and `Punctuation`.
* **unicode-perl** -
  Provide the data for supporting the Unicode-aware Perl character classes,
  corresponding to `\w`, `\s` and `\d`. This is also necessary for using
  Unicode-aware word boundary assertions. Note that if this feature is
  disabled, the `\s` and `\d` character classes are still available if the
  `unicode-bool` and `unicode-gencat` features are enabled, respectively.
* **unicode-script** -
  Provide the data for
  [Unicode scripts and script extensions](https://www.unicode.org/reports/tr24/).
  This includes, but is not limited to, `Arabic`, `Cyrillic`, `Hebrew`,
  `Latin` and `Thai`.
* **unicode-segment** -
  Provide the data necessary to provide the properties used to implement the
  [Unicode text segmentation algorithms](https://www.unicode.org/reports/tr29/).
  This enables using classes like `\p{gcb=Extend}`, `\p{wb=Katakana}` and
  `\p{sb=ATerm}`.


# Untrusted input

This crate can handle both untrusted regular expressions and untrusted
search text.

Untrusted regular expressions are handled by capping the size of a compiled
regular expression.
(See [`RegexBuilder::size_limit`](struct.RegexBuilder.html#method.size_limit).)
Without this, it would be trivial for an attacker to exhaust your system's
memory with expressions like `a{100}{100}{100}`.

Untrusted search text is allowed because the matching engine(s) in this
crate have time complexity `O(mn)` (with `m ~ regex` and `n ~ search
text`), which means there's no way to cause exponential blow-up like with
some other regular expression engines. (We pay for this by disallowing
features like arbitrary look-ahead and backreferences.)

When a DFA is used, pathological cases with exponential state blow-up are
avoided by constructing the DFA lazily or in an "online" manner. Therefore,
at most one new state can be created for each byte of input. This satisfies
our time complexity guarantees, but can lead to memory growth
proportional to the size of the input. As a stopgap, the DFA is only
allowed to store a fixed number of states. When the limit is reached, its
states are wiped and continues on, possibly duplicating previous work. If
the limit is reached too frequently, it gives up and hands control off to
another matching engine with fixed memory requirements.
(The DFA size limit can also be tweaked. See
[`RegexBuilder::dfa_size_limit`](struct.RegexBuilder.html#method.dfa_size_limit).)
*/

#![deny(missing_docs)]
#![cfg_attr(feature = "pattern", feature(pattern))]
#![warn(missing_debug_implementations)]

#[cfg(not(feature = "std"))]
compile_error!("`std` feature is currently required to build this crate");

// To check README's example
// TODO: Re-enable this once the MSRV is 1.43 or greater.
// See: https://github.com/rust-lang/regex/issues/684
// See: https://github.com/rust-lang/regex/issues/685
// #[cfg(doctest)]
// doc_comment::doctest!("../README.md");

#[cfg(feature = "std")]
pub use crate::error::Error;
#[cfg(feature = "std")]
pub use crate::re_builder::set_unicode::*;
#[cfg(feature = "std")]
pub use crate::re_builder::unicode::*;
#[cfg(feature = "std")]
pub use crate::re_set::unicode::*;
#[cfg(feature = "std")]
pub use crate::re_unicode::{
    escape, CaptureLocations, CaptureMatches, CaptureNames, Captures,
    Locations, Match, Matches, NoExpand, Regex, Replacer, ReplacerRef, Split,
    SplitN, SubCaptureMatches,
};

/**
Match regular expressions on arbitrary bytes.

This module provides a nearly identical API to the one found in the
top-level of this crate. There are two important differences:

1. Matching is done on `&[u8]` instead of `&str`. Additionally, `Vec<u8>`
is used where `String` would have been used.
2. Unicode support can be disabled even when disabling it would result in
matching invalid UTF-8 bytes.

# Example: match null terminated string

This shows how to find all null-terminated strings in a slice of bytes:

```rust
# use regex::bytes::Regex;
let re = Regex::new(r"(?-u)(?P<cstr>[^\x00]+)\x00").unwrap();
let text = b"foo\x00bar\x00baz\x00";

// Extract all of the strings without the null terminator from each match.
// The unwrap is OK here since a match requires the `cstr` capture to match.
let cstrs: Vec<&[u8]> =
    re.captures_iter(text)
      .map(|c| c.name("cstr").unwrap().as_bytes())
      .collect();
assert_eq!(vec![&b"foo"[..], &b"bar"[..], &b"baz"[..]], cstrs);
```

# Example: selectively enable Unicode support

This shows how to match an arbitrary byte pattern followed by a UTF-8 encoded
string (e.g., to extract a title from a Matroska file):

```rust
# use std::str;
# use regex::bytes::Regex;
let re = Regex::new(
    r"(?-u)\x7b\xa9(?:[\x80-\xfe]|[\x40-\xff].)(?u:(.*))"
).unwrap();
let text = b"\x12\xd0\x3b\x5f\x7b\xa9\x85\xe2\x98\x83\x80\x98\x54\x76\x68\x65";
let caps = re.captures(text).unwrap();

// Notice that despite the `.*` at the end, it will only match valid UTF-8
// because Unicode mode was enabled with the `u` flag. Without the `u` flag,
// the `.*` would match the rest of the bytes.
let mat = caps.get(1).unwrap();
assert_eq!((7, 10), (mat.start(), mat.end()));

// If there was a match, Unicode mode guarantees that `title` is valid UTF-8.
let title = str::from_utf8(&caps[1]).unwrap();
assert_eq!("☃", title);
```

In general, if the Unicode flag is enabled in a capture group and that capture
is part of the overall match, then the capture is *guaranteed* to be valid
UTF-8.

# Syntax

The supported syntax is pretty much the same as the syntax for Unicode
regular expressions with a few changes that make sense for matching arbitrary
bytes:

1. The `u` flag can be disabled even when disabling it might cause the regex to
match invalid UTF-8. When the `u` flag is disabled, the regex is said to be in
"ASCII compatible" mode.
2. In ASCII compatible mode, neither Unicode scalar values nor Unicode
character classes are allowed.
3. In ASCII compatible mode, Perl character classes (`\w`, `\d` and `\s`)
revert to their typical ASCII definition. `\w` maps to `[[:word:]]`, `\d` maps
to `[[:digit:]]` and `\s` maps to `[[:space:]]`.
4. In ASCII compatible mode, word boundaries use the ASCII compatible `\w` to
determine whether a byte is a word byte or not.
5. Hexadecimal notation can be used to specify arbitrary bytes instead of
Unicode codepoints. For example, in ASCII compatible mode, `\xFF` matches the
literal byte `\xFF`, while in Unicode mode, `\xFF` is a Unicode codepoint that
matches its UTF-8 encoding of `\xC3\xBF`. Similarly for octal notation when
enabled.
6. In ASCII compatible mode, `.` matches any *byte* except for `\n`. When the
`s` flag is additionally enabled, `.` matches any byte.

# Performance

In general, one should expect performance on `&[u8]` to be roughly similar to
performance on `&str`.
*/
#[cfg(feature = "std")]
pub mod bytes {
    pub use crate::re_builder::bytes::*;
    pub use crate::re_builder::set_bytes::*;
    pub use crate::re_bytes::*;
    pub use crate::re_set::bytes::*;
}

mod backtrack;
mod compile;
#[cfg(feature = "perf-dfa")]
mod dfa;
mod error;
mod exec;
mod expand;
mod find_byte;
mod input;
mod literal;
#[cfg(feature = "pattern")]
mod pattern;
mod pikevm;
mod pool;
mod prog;
mod re_builder;
mod re_bytes;
mod re_set;
mod re_trait;
mod re_unicode;
mod sparse;
mod utf8;

/// The `internal` module exists to support suspicious activity, such as
/// testing different matching engines and supporting the `regex-debug` CLI
/// utility.
#[doc(hidden)]
#[cfg(feature = "std")]
pub mod internal {
    pub use crate::compile::Compiler;
    pub use crate::exec::{Exec, ExecBuilder};
    pub use crate::input::{Char, CharInput, Input, InputAt};
    pub use crate::literal::LiteralSearcher;
    pub use crate::prog::{EmptyLook, Inst, InstRanges, Program};
}
