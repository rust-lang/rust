- Feature Name: read_exact and read_full
- Start Date: 2015-03-15
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Rust's `Write` trait has `write_all`, which is a convenience method that calls
`write` repeatedly to write an entire buffer. This proposal adds two similar
convenience methods to the `Read` trait: `read_full` and `read_exact`.
`read_full` calls `read` repeatedly until the buffer has been filled, EOF has
been reached, or an error other than `Interrupted` occurs. `read_exact` is
similar to `read_full`, except that reaching EOF before filling the buffer is
considered an error.

# Motivation

The `read` method may return fewer bytes than requested, and may fail with an
`Interrupted` error if a signal is received during the call. This requires
programs wishing to fill a buffer to call `read` repeatedly in a loop. This is
a very common need, and it would be nice if this functionality were provided in
the standard library. Many C and Rust programs have the same need, and solve it
in the same way. For example, Git has [`read_in_full`][git], which behaves like
the proposed `read_full`, and the Rust byteorder crate has
[`read_full`][byteorder], which behaves like the proposed `read_exact`.
[git]: https://github.com/git/git/blob/16da57c7c6c1fe92b32645202dd19657a89dd67d/wrapper.c#L246
[byteorder]: https://github.com/BurntSushi/byteorder/blob/2358ace61332e59f596c9006e1344c97295fdf72/src/new.rs#L184

# Detailed design

The following methods will be added to the `Read` trait:

``` rust
fn read_full(&mut self, buf: &mut [u8]) -> Result<usize>;
fn read_exact(&mut self, buf: &mut [u8]) -> Result<()>;
```

Additionally, default implementations of these methods will be provided:

``` rust
fn read_full(&mut self, mut buf: &mut [u8]) -> Result<usize> {
    let mut read = 0;
    while buf.len() > 0 {
        match self.read(buf) {
            Ok(0) => break,
            Ok(n) => { read += n; let tmp = buf; buf = &mut tmp[n..]; }
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(read)
}

fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
    if try!(self.read_full(buf)) != buf.len() {
        Err(Error::new(ErrorKind::UnexpectedEOF, "failed to fill whole buffer"))
    } else {
        Ok(())
    }
}
```

Finally, a new `ErrorKind::UnexpectedEOF` will be introduced, which will be
returned by `read_exact` in the event of a premature EOF.

# Drawbacks

Like `write_all`, these APIs are lossy: in the event of an error, there is no
way to determine the number of bytes that were successfully read before the
error. However, doing so would complicate the methods, and the caller will want
to simply fail if an error occurs the vast majority of the time. Situations
that require lower level control can still use `read` directly.

# Unanswered Questions

Naming. Is `read_full` the best name? Should `UnexpectedEOF` instead be
`ShortRead` or `ReadZero`?

# Alternatives

Use a more complicated return type to allow callers to retrieve the number of
bytes successfully read before an error occurred. As explained above, this
would complicate the use of these methods for very little gain. It's worth
noting that git's `read_in_full` is similarly lossy, and just returns an error
even if some bytes have been read.

Only provide `read_exact`, but parameterize the `UnexpectedEOF` or `ShortRead`
error kind with the number of bytes read to allow it to be used in place of
`read_full`. This would be less convenient to use in cases where EOF is not an
error.

Only provide `read_full`. This would cover most of the convenience (callers
could avoid the read loop), but callers requiring a filled buffer would have to
manually check if all of the desired bytes were read.

Finally, we could leave this out, and let every Rust user needing this
functionality continue to write their own `read_full` or `read_exact` function,
or have to track down an external crate just for one straightforward and
commonly used convenience method.
