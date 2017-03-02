- Feature Name: `ipaddr_octet_arrays`
- Start Date: 2016-02-12
- RFC PR: [rust-lang/rfcs#1498](https://github.com/rust-lang/rfcs/pull/1498)
- Rust Issue: [rust-lang/rust#32313](https://github.com/rust-lang/rust/issues/32313)

# Summary
[summary]: #summary

Add constructor and conversion functions for `std::net::Ipv6Addr` and
`std::net::Ipv4Addr` that are oriented around arrays of octets.

# Motivation
[motivation]: #motivation

Currently, the interface for `std::net::Ipv6Addr` is oriented around 16-bit
"segments".  The constructor takes eight 16-bit integers as arguments,
and the sole getter function, `segments`, returns an array of eight
16-bit integers.  This interface is unnatural when doing low-level network
programming, where IPv6 addresses are treated as a sequence of 16 octets.
For example, building and parsing IPv6 packets requires doing
bitwise arithmetic with careful attention to byte order in order to convert
between the on-wire format of 16 octets and the eight segments format used
by `std::net::Ipv6Addr`.

# Detailed design
[design]: #detailed-design

The following method would be added to `impl std::net::Ipv6Addr`:

```
pub fn octets(&self) -> [u8; 16] {
	self.inner.s6_addr
}
```

The following `From` trait would be implemented:

```
impl From<[u8; 16]> for Ipv6Addr {
	fn from(octets: [u8; 16]) -> Ipv6Addr {
		let mut addr: c::in6_addr = unsafe { std::mem::zeroed() };
		addr.s6_addr = octets;
		Ipv6Addr { inner: addr }
	}
}
```

For consistency, the following `From` trait would be
implemented for `Ipv4Addr`:

```
impl From<[u8; 4]> for Ipv4Addr {
	fn from(octets: [u8; 4]) -> Ipv4Addr {
		Ipv4Addr::new(octets[0], octets[1], octets[2], octets[3])
	}
}
```

Note: `Ipv4Addr` already has an `octets` method that returns a `[u8; 4]`.

# Drawbacks
[drawbacks]: #drawbacks

It adds additional functions to the API, which increases cognitive load
and maintenance burden.  That said, the functions are conceptually very simple
and their implementations short.

# Alternatives
[alternatives]: #alternatives

Do nothing.  The downside is that developers will need to resort to
bitwise arithmetic, which is awkward and error-prone (particularly with
respect to byte ordering) to convert between `Ipv6Addr` and the on-wire
representation of IPv6 addresses.  Or they will use their alternative
implementations of `Ipv6Addr`, fragmenting the ecosystem.

# Unresolved questions
[unresolved]: #unresolved-questions

