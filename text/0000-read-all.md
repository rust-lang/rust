- Feature Name: read_exact and ErrorKind::UnexpectedEOF
- Start Date: 2015-03-15
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Rust's `Write` trait has the `write_all` method, which is a convenience
method that writes a whole buffer, failing with `ErrorKind::WriteZero`
if the buffer cannot be written in full.

This RFC proposes adding its `Read` counterpart: a method (here called
`read_exact`) that reads a whole buffer, failing with an error (here
called `ErrorKind::UnexpectedEOF`) if the buffer cannot be read in full.

# Motivation

When dealing with serialization formats with fixed-length fields,
reading or writing less than the field's size is an error. For the
`Write` side, the `write_all` method does the job; for the `Read` side,
however, one has to call `read` in a loop until the buffer is completely
filled, or until a premature EOF is reached.

This leads to a profusion of similar helper functions. For instance, the
`byteorder` crate has a `read_full` function, and the `postgres` crate
has a `read_all` function. However, their handling of the premature EOF
condition differs: the `byteorder` crate has its own `Error` enum, with
`UnexpectedEOF` and `Io` variants, while the `postgres` crate uses an
`io::Error` with an `io::ErrorKind::Other`.

That can make it unnecessarily hard to mix uses of these helper
functions; for instance, if one wants to read a 20-byte tag (using one's
own helper function) followed by a big-endian integer, either the helper
function has to be written to use `byteorder::Error`, or the calling
code has to deal with two different ways to represent a premature EOF,
depending on which field encountered the EOF condition.

Additionally, when reading from an in-memory buffer, looping is not
necessary; it can be replaced by a size comparison followed by a
`copy_memory` (similar to `write_all` for `&mut [u8]`). If this
non-looping implementation is `#[inline]`, and the buffer size is known
(for instance, it's a fixed-size buffer in the stack, or there was an
earlier check of the buffer size against a larger value), the compiler
could potentially turn a read from the buffer followed by an endianness
conversion into the native endianness (as can happen when using the
`byteorder` crate) into a single-instruction direct load from the buffer
into a register.

# Detailed design

First, a new variant `UnexpectedEOF` is added to the `io::ErrorKind` enum.

The following method is added to the `Read` trait:

``` rust
fn read_exact(&mut self, buf: &mut [u8]) -> Result<()>;
```

Aditionally, a default implementation of this method is provided:

``` rust
fn read_exact(&mut self, mut buf: &mut [u8]) -> Result<()> {
    while buf.len() > 0 {
        match self.read(buf) {
            Ok(0) => break,
            Ok(n) => { let tmp = buf; buf = &mut tmp[n..]; }
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    if buf.len() > 0 {
        Err(Error::new(ErrorKind::UnexpectedEOF, "failed to fill whole buffer"))
    } else {
        Ok(())
    }
}
```

And an optimized implementation of this method for `&[u8]` is provided:

```rust
#[inline]
fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
    if (buf.len() > self.len()) {
        return Err(Error::new(ErrorKind::UnexpectedEOF, "failed to fill whole buffer"));
    }
    let (a, b) = self.split_at(buf.len());
    slice::bytes::copy_memory(a, buf);
    *self = b;
    Ok(())
}
```

# Naming

It's unfortunate that `write_all` used `WriteZero` for its `ErrorKind`;
were it named `UnexpectedEOF` (which is a much more intuitive name), the
same `ErrorKind` could be used for both functions.

The initial proposal for this `read_exact` method called it `read_all`,
for symmetry with `write_all`. However, that name could also be
interpreted as "read as many bytes as you can that fit on this buffer,
and return what you could read" instead of "read enough bytes to fill
this buffer, and fail if you couldn't read them all". The previous
discussion led to `read_exact` for the later meaning, and `read_full`
for the former meaning.

# Drawbacks

If this method fails, the buffer contents are undefined; the
`read_exact' method might have partially overwritten it. If the caller
requires "all-or-nothing" semantics, it must clone the buffer. In most
use cases, this is not a problem; the caller will discard or overwrite
the buffer in case of failure.

In the same way, if this method fails, there is no way to determine how
many bytes were read before it determined it couldn't completely fill
the buffer.

Situations that require lower level control can still use `read`
directly.

# Alternatives

The first alternative is to do nothing. Every Rust user needing this
functionality continues to write their own read_full or read_exact
function, or have to track down an external crate just for one
straightforward and commonly used convenience method. Additionally,
unless everybody uses the same external crate, every reimplementation of
this method will have slightly different error handling, complicating
mixing users of multiple copies of this convenience method.

The second alternative is to just add the `ErrorKind::UnexpectedEOF` or
similar. This would lead in the long run to everybody using the same
error handling for their version of this convenience method, simplifying
mixing their uses. However, it's questionable to add an `ErrorKind`
variant which is never used by the standard library.

Another alternative is to return the number of bytes read in the error
case. That makes the buffer contents defined also in the error case, at
the cost of increasing the size of the frequently-used `io::Error`
struct, for a rarely used return value. My objections to this
alternative are:

* If the caller has an use for the partially written buffer contents,
  then it's treating the "buffer partially filled" case as an
  alternative success case, not as a failure case. This is not a good
  match for the semantics of an `Err` return.
* Determining that the buffer cannot be completely filled can in some
  cases be much faster than doing a partial copy. Many callers are not
  going to be interested in an incomplete read, meaning that all the
  work of filling the buffer is wasted.
* As mentioned, it increases the size of a commonly used type in all
  cases, even when the code has no mention of `read_exact`.

The final alternative is `read_full`, which returns the number of bytes
read (`Result<usize>`) instead of failing. This means that every caller
has to check the return value against the size of the passed buffer, and
some are going to forget (or misimplement) the check. It also prevents
some optimizations (like the early return in case there will never be
enough data). There are, however, valid use cases for this alternative;
for instance, reading a file in fixed-size chunks, where the last chunk
(and only the last chunk) can be shorter. I believe this should be
discussed as a separate proposal; its pros and cons are distinct enough
from this proposal to merit its own arguments.

I believe that the case for `read_full` is weaker than `read_exact`, for
the following reasons:

* While `read_exact` needs an extra variant in `ErrorKind`, `read_full`
  has no new error cases. This means that implementing it yourself is
  easy, and multiple implementations have no drawbacks other than code
  duplication.
* While `read_exact` can be optimized with an early return in cases
  where the reader knows its total size (for instance, reading from a
  compressed file where the uncompressed size was given in a header),
  `read_full` has to always write to the output buffer, so there's not
  much to gain over a generic looping implementation calling `read`.

