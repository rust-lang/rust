- Feature Name: socket_timeouts
- Start Date: 2015-04-08
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add sockopt-style timeouts to `std::net` types.

# Motivation

Currently, operations on various socket types in `std::net` block
indefinitely (i.e., until the connection is closed or data is
transferred). But there are many contexts in which timing out a
blocking call is important.

The [goal of the current IO system][io-reform] is to gradually expose
cross-platform, blocking APIs for IO, especially APIs that directly
correspond to the underlying system APIs. Sockets are widely available
with nearly identical system APIs across the platforms Rust targets,
and this includes support for timeouts via [sockopts][sockopt].

So timeouts are well-motivated and well-suited to `std::net`.

# Detailed design

The proposal is to *directly expose* the timeout functionality
provided by [`setsockopt`][sockopt], in much the same way we currently
expose functionality like `set_nodelay`:

```rust
impl TcpStream {
    pub fn set_read_timeout(&self, dur: Duration) -> io::Result<()> { ... }
    pub fn set_write_timeout(&self, dur: Duration) -> io::Result<()> { ... }
}

impl UdpSocket {
    pub fn set_read_timeout(&self, dur: Duration) -> io::Result<()> { ... }
    pub fn set_write_timeout(&self, dur: Duration) -> io::Result<()> { ... }
}
```

These methods take an amount of time in the form of a `Duration`,
which is [undergoing stabilization][duration-reform]. They are
implemented via straightforward calls to `setsockopt`.

# Drawbacks

One potential downside to this design is that the timeouts are set
through direct mutation of the socket state, which can lead to
composition problems. For example, a socket could be passed to another
function which needs to use it with a timeout, but setting the timeout
clobbers any previous values. This lack of composability leads to
defensive programming in the form of "callee save" resets of timeouts,
for example. An alternative design is given below.

The advantage of binding the mutating APIs directly is that we keep a
close correspondence between the `std::net` types and their underlying
system types, and a close correspondence between Rust APIs and system
APIs. It's not clear that this kind of composability is important
enough in practice to justify a departure from the traditional API.

# Alternatives

A different approach would be to *wrap* socket types with a "timeout
modifier", which would be responsible for setting and resetting the
timeouts:

```rust
struct WithTimeout<T> {
    timeout: Duration,
    innter: T
}

impl<T> WithTimeout<T> {
    /// Returns the wrapped object, resetting the timeout
    pub fn into_inner(self) -> T { ... }
}

impl TcpStream {
    /// Wraps the stream with a timeout
    pub fn with_timeout(self, timeout: Duration) -> WithTimeout<TcpStream> { ... }
}

impl<T: Read> Read for WithTimeout<T> { ... }
impl<T: Write> Write for WithTimeout<T> { ... }
```

A [previous RFC][deadlines] spelled this out in more detail.

Unfortunately, such a "wrapping" API has problems of its own. It
creates unfortunate type incompatibilities, since you cannot store a
timeout-wrapped socket where a "normal" socket is expected.  It is
difficult to be "polymorphic" over timeouts.

Ultimately, it's not clear that the extra complexities of the type
distinction here are worth the better theoretical composability.

# Unresolved questions

Should we consider a preliminary version of this RFC that introduces
methods like `set_read_timeout_ms`, similar to `wait_timeout_ms` on
`Condvar`? These methods have been introduced elsewhere to provide a
stable way to use timeouts prior to `Duration` being stabilized.

[io-reform]: https://github.com/rust-lang/rfcs/blob/master/text/0517-io-os-reform.md
[sockopt]: http://pubs.opengroup.org/onlinepubs/009695399/functions/setsockopt.html
[duration-reform]: https://github.com/rust-lang/rfcs/pull/1040
[deadlines]: https://github.com/rust-lang/rfcs/pull/577/
