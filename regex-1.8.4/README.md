regex
=====
A Rust library for parsing, compiling, and executing regular expressions. Its
syntax is similar to Perl-style regular expressions, but lacks a few features
like look around and backreferences. In exchange, all searches execute in
linear time with respect to the size of the regular expression and search text.
Much of the syntax and implementation is inspired
by [RE2](https://github.com/google/re2).

[![Build status](https://github.com/rust-lang/regex/workflows/ci/badge.svg)](https://github.com/rust-lang/regex/actions)
[![Crates.io](https://img.shields.io/crates/v/regex.svg)](https://crates.io/crates/regex)
[![Rust](https://img.shields.io/badge/rust-1.60.0%2B-blue.svg?maxAge=3600)](https://github.com/rust-lang/regex)

### Documentation

[Module documentation with examples](https://docs.rs/regex).
The module documentation also includes a comprehensive description of the
syntax supported.

Documentation with examples for the various matching functions and iterators
can be found on the
[`Regex` type](https://docs.rs/regex/*/regex/struct.Regex.html).

### Usage

To bring this crate into your repository, either add `regex` to your
`Cargo.toml`, or run `cargo add regex`.

Here's a simple example that matches a date in YYYY-MM-DD format and prints the
year, month and day:

```rust
use regex::Regex;

fn main() {
    let re = Regex::new(r"(?x)
(?P<year>\d{4})  # the year
-
(?P<month>\d{2}) # the month
-
(?P<day>\d{2})   # the day
").unwrap();
    let caps = re.captures("2010-03-14").unwrap();

    assert_eq!("2010", &caps["year"]);
    assert_eq!("03", &caps["month"]);
    assert_eq!("14", &caps["day"]);
}
```

If you have lots of dates in text that you'd like to iterate over, then it's
easy to adapt the above example with an iterator:

```rust
use regex::Regex;

const TO_SEARCH: &'static str = "
On 2010-03-14, foo happened. On 2014-10-14, bar happened.
";

fn main() {
    let re = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();

    for caps in re.captures_iter(TO_SEARCH) {
        // Note that all of the unwraps are actually OK for this regex
        // because the only way for the regex to match is if all of the
        // capture groups match. This is not true in general though!
        println!("year: {}, month: {}, day: {}",
                 caps.get(1).unwrap().as_str(),
                 caps.get(2).unwrap().as_str(),
                 caps.get(3).unwrap().as_str());
    }
}
```

This example outputs:

```text
year: 2010, month: 03, day: 14
year: 2014, month: 10, day: 14
```

### Usage: Avoid compiling the same regex in a loop

It is an anti-pattern to compile the same regular expression in a loop since
compilation is typically expensive. (It takes anywhere from a few microseconds
to a few **milliseconds** depending on the size of the regex.) Not only is
compilation itself expensive, but this also prevents optimizations that reuse
allocations internally to the matching engines.

In Rust, it can sometimes be a pain to pass regular expressions around if
they're used from inside a helper function. Instead, we recommend using the
[`lazy_static`](https://crates.io/crates/lazy_static) crate to ensure that
regular expressions are compiled exactly once.

For example:

```rust,ignore
use regex::Regex;

fn some_helper_function(text: &str) -> bool {
    lazy_static! {
        static ref RE: Regex = Regex::new("...").unwrap();
    }
    RE.is_match(text)
}
```

Specifically, in this example, the regex will be compiled when it is used for
the first time. On subsequent uses, it will reuse the previous compilation.

### Usage: match regular expressions on `&[u8]`

The main API of this crate (`regex::Regex`) requires the caller to pass a
`&str` for searching. In Rust, an `&str` is required to be valid UTF-8, which
means the main API can't be used for searching arbitrary bytes.

To match on arbitrary bytes, use the `regex::bytes::Regex` API. The API
is identical to the main API, except that it takes an `&[u8]` to search
on instead of an `&str`. By default, `.` will match any *byte* using
`regex::bytes::Regex`, while `.` will match any *UTF-8 encoded Unicode scalar
value* using the main API.

This example shows how to find all null-terminated strings in a slice of bytes:

```rust
use regex::bytes::Regex;

let re = Regex::new(r"(?P<cstr>[^\x00]+)\x00").unwrap();
let text = b"foo\x00bar\x00baz\x00";

// Extract all of the strings without the null terminator from each match.
// The unwrap is OK here since a match requires the `cstr` capture to match.
let cstrs: Vec<&[u8]> =
    re.captures_iter(text)
      .map(|c| c.name("cstr").unwrap().as_bytes())
      .collect();
assert_eq!(vec![&b"foo"[..], &b"bar"[..], &b"baz"[..]], cstrs);
```

Notice here that the `[^\x00]+` will match any *byte* except for `NUL`. When
using the main API, `[^\x00]+` would instead match any valid UTF-8 sequence
except for `NUL`.

### Usage: match multiple regular expressions simultaneously

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

### Usage: enable SIMD optimizations

SIMD optimizations are enabled automatically on Rust stable 1.27 and newer.
For nightly versions of Rust, this requires a recent version with the SIMD
features stabilized.


### Usage: a regular expression parser

This repository contains a crate that provides a well tested regular expression
parser, abstract syntax and a high-level intermediate representation for
convenient analysis. It provides no facilities for compilation or execution.
This may be useful if you're implementing your own regex engine or otherwise
need to do analysis on the syntax of a regular expression. It is otherwise not
recommended for general use.

[Documentation `regex-syntax`.](https://docs.rs/regex-syntax)


### Crate features

This crate comes with several features that permit tweaking the trade off
between binary size, compilation time and runtime performance. Users of this
crate can selectively disable Unicode tables, or choose from a variety of
optimizations performed by this crate to disable.

When all of these features are disabled, runtime match performance may be much
worse, but if you're matching on short strings, or if high performance isn't
necessary, then such a configuration is perfectly serviceable. To disable
all such features, use the following `Cargo.toml` dependency configuration:

```toml
[dependencies.regex]
version = "1.3"
default-features = false
# regex currently requires the standard library, you must re-enable it.
features = ["std"]
```

This will reduce the dependency tree of `regex` down to a single crate
(`regex-syntax`).

The full set of features one can disable are
[in the "Crate features" section of the documentation](https://docs.rs/regex/*/#crate-features).


### Minimum Rust version policy

This crate's minimum supported `rustc` version is `1.60.0`.

The current **tentative** policy is that the minimum Rust version required
to use this crate can be increased in minor version updates. For example, if
regex 1.0 requires Rust 1.20.0, then regex 1.0.z for all values of `z` will
also require Rust 1.20.0 or newer. However, regex 1.y for `y > 0` may require a
newer minimum version of Rust.

In general, this crate will be conservative with respect to the minimum
supported version of Rust.


### License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   https://opensource.org/licenses/MIT)

at your option.

The data in `regex-syntax/src/unicode_tables/` is licensed under the Unicode
License Agreement
([LICENSE-UNICODE](https://www.unicode.org/copyright.html#License)).
