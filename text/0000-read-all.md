- Feature Name: read_exact and read_full
- Start Date: 2015-03-15
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Rust's Write trait has write_all, which attempts to write an entire
buffer.  This proposal adds two new methods, read_full and read_exact.
read_full attempts to read a fixed number of bytes into a given
buffer, and returns Ok(n) if it succeeds or in the event of EOF.
read_exact attempts to read a fixed number of bytes into a given
buffer, and returns Ok(n) if it succeeds and Err(ErrorKind::ShortRead)
if it fails.

# Motivation

The new read_exact method will allow programs to read from disk
without having to write their own read loops to handle EINTR.  Most
Rust programs which need to read from disk will prefer this to the
plain read function.  Many C programs have the same need, and solve it
the same way (e.g. git has read_in_full).  Here's one example of a
Rust library doing this:
https://github.com/BurntSushi/byteorder/blob/master/src/new.rs#L184

The read_full method is useful the common case of implementing
buffered reads from a file or socket.  In this case, a short read due
to EOF is an expected outcome, and the caller must check the number of
bytes returned.

# Detailed design

The read_full function will take a mutable, borrowed slice of u8 to
read into, and will attempt to fill that entire slice with data.

It will loop, calling read() once per iteration and attempting to read
the remaining amount of data.  If read returns EINTR, the loop will
retry.  If there are no more bytes to read (as signalled by a return
of Ok(0) from read()), the number of bytes read so far
will be returned.  In the event of another error, that error will be
returned. After a read call returns having successfully read some
bytes, the total number of bytes read will be updated.  If that total
is equal to the size of the buffer, read_full will return successfully.

The read_exact method can be implemented in terms of read_full.

# Drawbacks

The major weakness of this API (shared with write_all) is that in the
event of an error, there is no way to return the number of bytes that
were successfully read before the error.  But returning that data
would require a much more complicated return type, as well as
requiring more work on the part of callers.

# Alternatives

One alternative design would return some new kind of Result which
could report the number of bytes sucessfully read before an error.

If we wanted one method instead of two, ErrorKind::ShortRead could be
parameterized with the number of bytes read before EOF.  But this
would increase the size of ErrorKind.

Or we could leave this out, and let every Rust user write their own
read_full or read_exact function, or import a crate of stuff just for
this one function.
