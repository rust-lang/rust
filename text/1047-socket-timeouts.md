- Feature Name: `socket_timeouts`
- Start Date: 2015-04-08
- RFC PR: [rust-lang/rfcs#1047](https://github.com/rust-lang/rfcs/pull/1047)
- Rust Issue: [rust-lang/rust#25619](https://github.com/rust-lang/rust/issues/25619)

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
    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> { ... }
    pub fn read_timeout(&self) -> io::Result<Option<Duration>>;

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> { ... }
    pub fn write_timeout(&self) -> io::Result<Option<Duration>>;
}

impl UdpSocket {
    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> { ... }
    pub fn read_timeout(&self) -> io::Result<Option<Duration>>;

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> { ... }
    pub fn write_timeout(&self) -> io::Result<Option<Duration>>;
}
```

The setter methods take an amount of time in the form of a `Duration`,
which is [undergoing stabilization][duration-reform]. They are
implemented via straightforward calls to `setsockopt`. The `Option` is
used to signify no timeout (for both setting and
getting). Consequently, `Some(Duration::new(0, 0))` is a possible
argument; the setter methods will return an IO error of kind
`InvalidInput` in this case. (See Alternatives for other approaches.)

The corresponding socket options are `SO_RCVTIMEO` and `SO_SNDTIMEO`.

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

## Taking `Duration` directly

Using an `Option<Duration>` introduces a certain amount of complexity
-- it raises the issue of `Some(Duration::new(0, 0))`, and it's
slightly more verbose to set a timeout.

An alternative would be to take a `Duration` directly, and interpret a
zero length duration as "no timeout" (which is somewhat traditional in
C APIs). That would make the API somewhat more familiar, but less
Rustic, and it becomes somewhat easier to pass in a zero value by
accident (without thinking about this possibility).

Note that both styles of API require code that does arithmetic on
durations to check for zero in advance.

Aside from fitting Rust idioms better, the main proposal also gives a
somewhat stronger indication of a bug when things go wrong (rather
than simply failing to time out, for example).

## Combining with nonblocking support

Another possibility would be to provide a single method that can
choose between blocking indefinitely, blocking with a timeout, and
nonblocking mode:

```rust
enum BlockingMode {
    Nonblocking,
    Blocking,
    Timeout(Duration)
}
```

This `enum` makes clear that it doesn't make sense to have both a
timeout and put the socket in nonblocking mode. On the other hand, it
would relinquish the one-to-one correspondence between Rust
configuration APIs and underlying socket options.

## Wrapping for compositionality

A different approach would be to *wrap* socket types with a "timeout
modifier", which would be responsible for setting and resetting the
timeouts:

```rust
struct WithTimeout<T> {
    timeout: Duration,
    inner: T
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
