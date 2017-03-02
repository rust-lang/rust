- Feature Name: `net2_mutators`
- Start Date: 2016-01-12
- RFC PR: [rust-lang/rfcs#1461](https://github.com/rust-lang/rfcs/pull/1461)
- Rust Issue: [rust-lang/rust#31766](https://github.com/rust-lang/rust/issues/31766)

# Summary
[summary]: #summary

[RFC 1158](https://github.com/rust-lang/rfcs/pull/1158) proposed the addition
of more functionality for the `TcpStream`, `TcpListener` and `UdpSocket` types,
but was declined so that those APIs could be built up out of tree in the [net2
crate](https://crates.io/crates/net2/). This RFC proposes pulling portions of
net2's APIs into the standard library.

# Motivation
[motivation]: #motivation

The functionality provided by the standard library's wrappers around standard
networking types is fairly limited, and there is a large set of well supported,
standard functionality that is not currently implemented in `std::net` but has
existed in net2 for some time.

All of the methods to be added map directly to equivalent system calls.

This does not cover the entirety of net2's APIs. In particular, this RFC does
not propose to touch the builder types.

# Detailed design
[design]: #detailed-design

The following methods will be added:

```rust
impl TcpStream {
    fn set_nodelay(&self, nodelay: bool) -> io::Result<()>;
    fn nodelay(&self) -> io::Result<bool>;

    fn set_ttl(&self, ttl: u32) -> io::Result<()>;
    fn ttl(&self) -> io::Result<u32>;

    fn set_only_v6(&self, only_v6: bool) -> io::Result<()>;
    fn only_v6(&self) -> io::Result<bool>;

    fn take_error(&self) -> io::Result<Option<io::Error>>;

    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()>;
}

impl TcpListener {
    fn set_ttl(&self, ttl: u32) -> io::Result<()>;
    fn ttl(&self) -> io::Result<u32>;

    fn set_only_v6(&self, only_v6: bool) -> io::Result<()>;
    fn only_v6(&self) -> io::Result<bool>;

    fn take_error(&self) -> io::Result<Option<io::Error>>;

    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()>;
}

impl UdpSocket {
    fn set_broadcast(&self, broadcast: bool) -> io::Result<()>;
    fn broadcast(&self) -> io::Result<bool>;

    fn set_multicast_loop_v4(&self, multicast_loop_v4: bool) -> io::Result<()>;
    fn multicast_loop_v4(&self) -> io::Result<bool>;

    fn set_multicast_ttl_v4(&self, multicast_ttl_v4: u32) -> io::Result<()>;
    fn multicast_ttl_v4(&self) -> io::Result<u32>;

    fn set_multicast_loop_v6(&self, multicast_loop_v6: bool) -> io::Result<()>;
    fn multicast_loop_v6(&self) -> io::Result<bool>;

    fn set_ttl(&self, ttl: u32) -> io::Result<()>;
    fn ttl(&self) -> io::Result<u32>;

    fn set_only_v6(&self, only_v6: bool) -> io::Result<()>;
    fn only_v6(&self) -> io::Result<bool>;

    fn join_multicast_v4(&self, multiaddr: &Ipv4Addr, interface: &Ipv4Addr) -> io::Result<()>;
    fn join_multicast_v6(&self, multiaddr: &Ipv6Addr, interface: u32) -> io::Result<()>;

    fn leave_multicast_v4(&self, multiaddr: &Ipv4Addr, interface: &Ipv4Addr) -> io::Result<()>;
    fn leave_multicast_v6(&self, multiaddr: &Ipv6Addr, interface: u32) -> io::Result<()>;

    fn connect<A: ToSocketAddrs>(&self, addr: A) -> Result<()>;
    fn send(&self, buf: &[u8]) -> Result<usize>;
    fn recv(&self, buf: &mut [u8]) -> Result<usize>;

    fn take_error(&self) -> io::Result<Option<io::Error>>;

    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()>;
}
```

The traditional approach would be to add these as unstable, inherent methods.
However, since inherent methods take precedence over trait methods, this would
cause all code using the extension traits in net2 to start reporting stability
errors. Instead, we have two options:

1. Add this functionality as *stable* inherent methods. The rationale here would
    be that time in a nursery crate acts as a de facto stabilization period.
2. Add this functionality via *unstable* extension traits. When/if we decide to
    stabilize, we would deprecate the trait and add stable inherent methods.
    Extension traits are a bit more annoying to work with, but this would give
    us a formal stabilization period.

Option 2 seems like the safer approach unless people feel comfortable with these
APIs.

# Drawbacks
[drawbacks]: #drawbacks

This is a fairly significant increase in the surface areas of these APIs, and
most users will never touch some of the more obscure functionality that these
provide.

# Alternatives
[alternatives]: #alternatives

We can leave some or all of this functionality in net2.

# Unresolved questions
[unresolved]: #unresolved-questions

The stabilization path (see above).
