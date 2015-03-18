- Feature Name: read_all
- Start Date: 2015-03-15
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Rust's Write trait has write_all, which attempts to write an entire
buffer.  This proposal adds read_all, which attempts to read a fixed
number of bytes into a given buffer.

# Motivation

The new read_all method will allow programs to read from disk without
having to write their own read loops.  Most Rust programs which need
to read from disk will prefer this to the plain read function.  Many C
programs have the same need, and solve it the same way (e.g. git has
read_in_full).  Here's one example of a Rust library doing this:
https://github.com/BurntSushi/byteorder/blob/master/src/new.rs#L184

# Detailed design

The read_all function will take a mutable, borrowed slice of u8 to
read into, and will attempt to fill that entire slice with data.

It will loop, calling read() once per iteration and attempting to read
the remaining amount of data.  If read returns EINTR, the loop will
retry.  If there are no more bytes to read (as signalled by a return
of Ok(0) from read()), a new error type, ErrorKind::ShortRead, will be
returned.  In the event of another error, that error will be
returned. After a read call returns having successfully read some
bytes, the total number of bytes read will be updated.  If that
total is equal to the size of the buffer, read will return
successfully.

# Drawbacks

The major weakness of this API (shared with write_all) is that in the
event of an error, there is no way to return the number of bytes that
were successfully read before the error.  But since that is the design
of write_all, it makes sense to mimic that design decision for read_all.

# Alternatives

One alternative design would return some new kind of Result which
could report the number of bytes sucessfully read before an error.
This would be inconsistent with write_all, but arguably more correct.

Another would be that ErrorKind::ShortRead would be parameterized by
the number of bytes read before EOF.  The downside of this is that it
bloats the size of io::Error.

Finally, in the event of a short read, we could return Ok(number of
bytes read before EOF) instead of an error.  But then every user would
have to check for this case.  And it would be inconsistent with
write_all.

Or we could leave this out, and let every Rust user write their own
read_all function -- like savages.
