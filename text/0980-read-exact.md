- Feature Name: read_exact
- Start Date: 2015-03-15
- RFC PR: https://github.com/rust-lang/rfcs/pull/980
- Rust Issue: https://github.com/rust-lang/rust/issues/27585

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
    while !buf.is_empty() {
        match self.read(buf) {
            Ok(0) => break,
            Ok(n) => { let tmp = buf; buf = &mut tmp[n..]; }
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    if !buf.is_empty() {
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

The detailed semantics of `read_exact` are as follows: `read_exact`
reads exactly the number of bytes needed to completely fill its `buf`
parameter. If that's not possible due to an "end of file" condition
(that is, the `read` method would return 0 even when passed a buffer
with at least one byte), it returns an `ErrorKind::UnexpectedEOF` error.

On success, the read pointer is advanced by the number of bytes read, as
if the `read` method had been called repeatedly to fill the buffer. On
any failure (including an `ErrorKind::UnexpectedEOF`), the read pointer
might have been advanced by any number between zero and the number of
bytes requested (inclusive), and the contents of its `buf` parameter
should be treated as garbage (any part of it might or might not have
been overwritten by unspecified data).

Even if the failure was an `ErrorKind::UnexpectedEOF`, the read pointer
might have been advanced by a number of bytes less than the number of
bytes which could be read before reaching an "end of file" condition.

The `read_exact` method will never return an `ErrorKind::Interrupted`
error, similar to the `read_to_end` method.

Similar to the `read` method, no guarantees are provided about the
contents of `buf` when this function is called; implementations cannot
rely on any property of the contents of `buf` being true. It is
recommended that implementations only write data to `buf` instead of
reading its contents.

# About ErrorKind::Interrupted

Whether or not `read_exact` can return an `ErrorKind::Interrupted` error
is orthogonal to its semantics. One could imagine an alternative design
where `read_exact` could return an `ErrorKind::Interrupted` error.

The reason `read_exact` should deal with `ErrorKind::Interrupted` itself
is its non-idempotence. On failure, it might have already partially
advanced its read pointer an unknown number of bytes, which means it
can't be easily retried after an `ErrorKind::Interrupted` error.

One could argue that it could return an `ErrorKind::Interrupted` error
if it's interrupted before the read pointer is advanced. But that
introduces a non-orthogonality in the design, where it might either
return or retry depending on whether it was interrupted at the beginning
or in the middle. Therefore, the cleanest semantics is to always retry.

There's precedent for this choice in the `read_to_end` method. Users who
need finer control should use the `read` method directly.

# About the read pointer

This RFC proposes a `read_exact` function where the read pointer
(conceptually, what would be returned by `Seek::seek` if the stream was
seekable) is unspecified on failure: it might not have advanced at all,
have advanced in full, or advanced partially.

Two possible alternatives could be considered: never advance the read
pointer on failure, or always advance the read pointer to the "point of
error" (in the case of `ErrorKind::UnexpectedEOF`, to the end of the
stream).

Never advancing the read pointer on failure would make it impossible to
have a default implementation (which calls `read` in a loop), unless the
stream was seekable. It would also impose extra costs (like creating a
temporary buffer) to allow "seeking back" for non-seekable streams.

Always advancing the read pointer to the end on failure is possible; it
happens without any extra code in the default implementation. However,
it can introduce extra costs in optimized implementations. For instance,
the implementation given above for `&[u8]` would need a few more
instructions in the error case. Some implementations (for instance,
reading from a compressed stream) might have a larger extra cost.

The utility of always advancing the read pointer to the end is
questionable; for non-seekable streams, there's not much that can be
done on an "end of file" condition, so most users would discard the
stream in both an "end of file" and an `ErrorKind::UnexpectedEOF`
situation. For seekable streams, it's easy to seek back, but most users
would treat an `ErrorKind::UnexpectedEOF` as a "corrupted file" and
discard the stream anyways.

Users who need finer control should use the `read` method directly, or
when available use the `Seek` trait.

# About the buffer contents

This RFC proposes that the contents of the output buffer be undefined on
an error return. It might be untouched, partially overwritten, or
completely overwritten (even if less bytes could be read; for instance,
this method might in theory use it as a scratch space).

Two possible alternatives could be considered: do not touch it on
failure, or overwrite it with valid data as much as possible.

Never touching the output buffer on failure would make it much more
expensive for the default implementation (which calls `read` in a loop),
since it would have to read into a temporary buffer and copy to the
output buffer on success. Any implementation which cannot do an early
return for all failure cases would have similar extra costs.

Overwriting as much as possible with valid data makes some sense; it
happens without any extra cost in the default implementation. However,
for optimized implementations this extra work is useless; since the
caller can't know how much is valid data and how much is garbage, it
can't make use of the valid data.

Users who need finer control should use the `read` method directly.

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
