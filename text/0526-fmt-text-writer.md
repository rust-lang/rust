- Start Date: 2014-12-30
- RFC PR: https://github.com/rust-lang/rfcs/pull/526
- Rust Issue: https://github.com/rust-lang/rust/issues/20352

# Summary

Statically enforce that the `std::fmt` module can only create valid UTF-8 data
by removing the arbitrary `write` method in favor of a `write_str` method.

# Motivation

Today it is conventionally true that the output from macros like `format!` and
well as implementations of `Show` only create valid UTF-8 data. This is not
statically enforced, however. As a consequence the `.to_string()` method must
perform a `str::is_utf8` check before returning a `String`.

This `str::is_utf8` check is currently [one of the most costly parts][bench1]
of the formatting subsystem while normally just being a redundant check.

[bench1]: https://gist.github.com/alexcrichton/162a5f8f93062800c914

Additionally, it is possible to statically enforce the convention that `Show`
only deals with valid unicode, and as such the possibility of doing so should be
explored.

# Detailed design

The `std::fmt::FormatWriter` trait will be redefined as:

```rust
pub trait Writer {
    fn write_str(&mut self, data: &str) -> Result;
    fn write_char(&mut self, ch: char) -> Result {
        // default method calling write_str
    }
    fn write_fmt(&mut self, f: &Arguments) -> Result {
        // default method calling fmt::write
    }
}
```

There are a few major differences with today's trait:

* The name has changed to `Writer` in accordance with [RFC 356][rfc356]
* The `write` method has moved from taking `&[u8]` to taking `&str` instead.
* A `write_char` method has been added.

[rfc356]: https://github.com/rust-lang/rfcs/blob/master/text/0356-no-module-prefixes.md

The corresponding methods on the `Formatter` structure will also be altered to
respect these signatures.

The key idea behind this API is that the `Writer` trait only operates on unicode
data. The `write_str` method is a static enforcement of UTF-8-ness, and using
`write_char` follows suit as a `char` can only be a valid unicode codepoint.

With this trait definition, the implementation of `Writer` for `Vec<u8>` will be
removed (note this is *not* the `io::Writer` implementation) in favor of an
implementation directly on `String`. The `.to_string()` method will change
accordingly (as well as `format!`) to write directly into a `String`, bypassing
all UTF-8 validity checks afterwards.

This change [has been implemented][branch] in a branch of mine, and as expected
the [benchmark numbers have improved][bench2] for the much larger texts.

[branch]: https://github.com/alexcrichton/rust/tree/fmt-text
[bench2]: https://gist.github.com/alexcrichton/182ccef5d8c2583a2423

Note that a key point of the changes implemented is that a call to `write!` into
an arbitrary `io::Writer` is *still valid* as it's still just a sink for bytes.
The changes outlined in this RFC will only affect `Show` and other formatting
trait implementations. As can be seen from the sample implementation, the
fallout is quite minimal with respect to the rest of the standard library.

# Drawbacks

A version of this RFC has been [previously postponed][rfc57], but this variant
is much less ambitious in terms of generic `TextWriter` support. At this time
the design of `fmt::Writer` is purposely conservative.

[rfc57]: https://github.com/rust-lang/rfcs/pull/57

There are currently some use cases today where a `&mut Formatter` is interpreted
as a `&mut Writer`, e.g. for the `Show` impl of `Json`. This is undoubtedly used
outside this repository, and it would break all of these users relying on the
binary functionality of the old `FormatWriter`.

# Alternatives

Another possible solution to specifically the performance problem is to have an
`unsafe` flag on a `Formatter` indicating that only valid utf-8 data was
written, and if all sub-parts of formatting set this flag then the data can be
assumed utf-8. In general relying on `unsafe` apis is less "pure" than relying
on the type system instead.

The `fmt::Writer` trait can also be located as `io::TextWriter` instead to
emphasize its possible future connection with I/O, although there are not
concrete plans today to develop these connections.

# Unresolved questions

* It is unclear to what degree a `fmt::Writer` needs to interact with
  `io::Writer` and the various adaptors/buffers. For example one would have to
  implement their own `BufferedWriter` for a `fmt::Writer`.
