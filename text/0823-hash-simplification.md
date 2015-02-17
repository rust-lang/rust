- Feature Name: hash
- Start Date: 2015-02-17
- RFC PR: https://github.com/rust-lang/rfcs/pull/823
- Rust Issue: https://github.com/rust-lang/rust/issues/22467

# Summary

Pare back the `std::hash` module's API to improve ergonomics of usage and
definitions. While an alternative scheme more in line with what Java and C++
have is considered, the current `std::hash` module will remain largely as-is
with modifications to its core traits.

# Motivation

There are a number of motivations for this RFC, and each will be explained in
term.

## API ergonomics

Today the API of the `std::hash` module is sometimes considered overly
complicated and it may not be pulling its weight. As a recap, the API looks
like:

```rust
trait Hash<H: Hasher> {
    fn hash(&self, state: &mut H);
}
trait Hasher {
    type Output;
    fn reset(&mut self);
    fn finish(&self) -> Self::Output;
}
trait Writer {
    fn write(&mut self, data: &[u8]);
}
```

The `Hash` trait is implemented by various types where the `H` type parameter
signifies the hashing algorithm that the `impl` block corresponds to. Each
`Hasher` is opaque when taken generically and is frequently paired with a bound
of `Writer` to allow feeding in arbitrary bytes.

The purpose of not having a `Writer` supertrait on `Hasher` or on the `H` type
parameter is to allow hashing algorithms that are *not* byte-stream oriented
(e.g. Java-like algorithms). Unfortunately all primitive types in Rust are only
defined for `Hash<H> where H: Writer + Hasher`, essentially forcing a
byte-stream oriented hashing algorithm for all hashing.

Some examples of using this API are:

```rust
use std::hash::{Hash, Hasher, Writer, SipHasher};

impl<S: Hasher + Writer> Hash<S> for MyType {
    fn hash(&self, s: &mut S) {
        self.field1.hash(s);
        // don't want to hash field2
        self.field3.hash(s);
    }
}

fn sip_hash<T: Hash<SipHasher>>(t: &T) -> u64 {
    let mut s = SipHasher::new_with_keys(0, 0);
    t.hash(&mut s);
    s.finish()
}
```

Forcing many `impl` blocks to require `Hasher + Writer` becomes onerous over
times and also requires at least 3 imports for a custom implementation of
`hash`. Taking a generically hashable `T` is also somewhat cumbersome,
especially if the hashing algorithm isn't known in advance.

Overall the `std::hash` API is generic enough that its usage is somewhat verbose
and becomes tiresome over time to work with. This RFC strives to make this API
easier to work with.

## Forcing byte-stream oriented hashing

Much of the `std::hash` API today is oriented around hashing a stream of bytes
(blocks of `&[u8]`). This is not a hard requirement by the API (discussed
above), but in practice this is essentially what happens everywhere. This form
of hashing is not always the most efficient, although it is often one of the
more flexible forms of hashing.

Other languages such as Java and C++ have a hashing API that looks more like:

```rust
trait Hash {
    fn hash(&self) -> usize;
}
```

This expression of hashing is not byte-oriented but is also much less generic
(an algorithm for hashing is predetermined by the type itself). This API is
encodable with today's traits as:

```rust
struct Slot(u64);

impl Hash<Slot> for MyType {
    fn hash(&self, slot: &mut Slot) {
        *slot = Slot(self.precomputed_hash);
    }
}

impl Hasher for Slot {
    type Output = u64;
    fn reset(&mut self) { *self = Slot(0); }
    fn finish(&self) -> u64 { self.0 }
}
```

This form of hashing (which is useful for performance sometimes) is difficult to
work with primarily because of the frequent bounds on `Writer` for hashing.

## Non-applicability for well-known hashing algorithms

One of the current aspirations for the `std::hash` module was to be appropriate
for hashing algorithms such as MD5, SHA\*, etc. The current API has proven
inadequate, however, for the primary reason of hashing being so generic. For
example it should in theory be possible to calculate the SHA1 hash of a byte
slice via:

```rust
let data: &[u8] = ...;
let hash = std::hash::hash::<&[u8], Sha1>(data);
```

There are a number of pitfalls to this approach:

* Due to slices being able to be hashed generically, each byte will be written
  individually to the `Sha1` state, which is likely to not be very efficient.
* Due to slices being able to be hashed generically, the length of the slice is
  first written to the `Sha1` state, which is likely not desired.

The key observation is that the hash values produced in a Rust program are
**not** reproducible outside of Rust. For this reason, APIs for reproducible
hashes to be verified elsewhere will explicitly not be considered in the design
for `std::hash`. It is expected that an external crate may wish to provide a
trait for these hashing algorithms and it would not be bounded by
`std::hash::Hash`, but instead perhaps a "byte container" of some form.

# Detailed design

This RFC considers two possible designs as a replacement of today's `std::hash`
API. One is a "minor refactoring" of the current API while the
other is a much more radical change towards being conservative. This section
will propose the minor refactoring change and the other may be found in the
[Alternatives](#alternatives) section.

## API

The new API of `std::hash` would be:

```rust
trait Hash {
    fn hash<H: Hasher>(&self, h: &mut H);

    fn hash_slice<H: Hasher>(data: &[Self], h: &mut H) {
        for piece in data {
            data.hash(h);
        }
    }
}

trait Hasher {
    fn write(&mut self, data: &[u8]);
    fn finish(&self) -> u64;

    fn write_u8(&mut self, i: u8) { ... }
    fn write_i8(&mut self, i: i8) { ... }
    fn write_u16(&mut self, i: u16) { ... }
    fn write_i16(&mut self, i: i16) { ... }
    fn write_u32(&mut self, i: u32) { ... }
    fn write_i32(&mut self, i: i32) { ... }
    fn write_u64(&mut self, i: u64) { ... }
    fn write_i64(&mut self, i: i64) { ... }
    fn write_usize(&mut self, i: usize) { ... }
    fn write_isize(&mut self, i: isize) { ... }
}
```

This API is quite similar to today's API, but has a few tweaks:

* The `Writer` trait has been removed by folding it directly into the `Hasher`
  trait. As part of this movement the `Hasher` trait grew a number of
  specialized `write_foo` methods which the primitives will call. This should
  help regain some performance losses where forcing a byte-oriented stream is
  a performance loss.

* The `Hasher` trait no longer has a `reset` method.

* The `Hash` trait's type parameter is on the *method*, not on the trait. This
  implies that the trait is no longer object-safe, but it is much more ergonomic
  to operate over generically.

* The `Hash` trait now has a `hash_slice` method to slice a number of instances
  of `Self` at once. This will allow optimization of the `Hash` implementation
  of `&[u8]` to translate to a raw `write` as well as other various slices of
  primitives.

* The `Output` associated type was removed in favor of an explicit `u64` return
  from `finish`.

The purpose of this API is to continue to allow APIs to be generic over the
hashing algorithm used. This would allow `HashMap` continue to use a randomly
keyed SipHash as its default algorithm (e.g. continuing to provide DoS
protection, more information on this below). An example encoding of the
alternative API (proposed below) would look like:

```rust
impl Hasher for u64 {
    fn write(&mut self, data: &[u8]) {
        for b in data.iter() { self.write_u8(*b); }
    }
    fn finish(&self) -> u64 { *self }

    fn write_u8(&mut self, i: u8) { *self = combine(*self, i); }
    // and so on...
}
```

## `HashMap` and `HashState`

For both this recommendation as well as the alternative below, this RFC proposes
removing the `HashState` trait and `Hasher` structure (as well as the
`hash_state` module) in favor of the following API:

```rust
struct HashMap<K, V, H = DefaultHasher>;

impl<K: Eq + Hash, V> HashMap<K, V> {
    fn new() -> HashMap<K, V, DefaultHasher> {
        HashMap::with_hasher(DefaultHasher::new())
    }
}

impl<K: Eq, V, H: Fn(&K) -> u64> HashMap<K, V, H> {
    fn with_hasher(hasher: H) -> HashMap<K, V, H>;
}

impl<K: Hash> Fn(&K) -> u64 for DefaultHasher {
    fn call(&self, arg: &K) -> u64 {
        let (k1, k2) = self.siphash_keys();
        let mut s = SipHasher::new_with_keys(k1, k2);
        arg.hash(&mut s);
        s.finish()
    }
}
```

The precise details will be affected based on which design in this RFC is
chosen, but the general idea is to move from a custom trait to the standard `Fn`
trait for calculating hashes.

# Drawbacks

* This design is a departure from the precedent set by many other languages. In
  doing so, however, it is arguably easier to implement `Hash` as it's more
  obvious how to feed in incremental state. We also do not lock ourselves into a
  particular hashing algorithm in case we need to alternate in the future.

* Implementations of `Hash` cannot be specialized and are forced to operate
  generically over the hashing algorithm provided. This may cause a loss of
  performance in some cases. Note that this could be remedied by moving the type
  parameter to the trait instead of the method, but this would lead to a loss in
  ergonomics for generic consumers of `T: Hash`.

* Manual implementations of `Hash` are somewhat cumbersome still by requiring a
  separate `Hasher` parameter which is not necessarily always desired.

* The API of `Hasher` is approaching the realm of serialization/reflection and
  it's unclear whether its API should grow over time to support more basic Rust
  types. It would be unfortunate if the `Hasher` trait approached a full-blown
  `Encoder` trait (as `rustc-serialize` has).

# Alternatives

As alluded to in the "Detailed design" section the primary alternative to this
RFC, which still improves ergonomics, is to remove the generic-ness over the
hashing algorithm.

## API

The new API of `std::hash` would be:

```rust
trait Hash {
    fn hash(&self) -> usize;
}

fn combine(a: usize, b: usize) -> usize;
```

The `Writer`, `Hasher`, and `SipHasher` structures/traits would all be removed
from `std::hash`. This definition is more or less the Rust equivalent of the
Java/C++ hashing infrastructure. This API is a vast simplification of what
exists today and allows implementations of `Hash` as well as consumers of `Hash`
to quite ergonomically work with hash values as well as hashable objects.

> **Note**: The choice of `usize` instead of `u64` reflects [C++'s
> choice][cpp-hash] here as well, but it is quite easy to use one instead of
> the other.

## Hashing algorithm

With this definition of `Hash`, each type must pre-ordain a particular hash
algorithm that it implements. Using an alternate algorithm would require a
separate newtype wrapper.

Most implementations would still use `#[derive(Hash)]` which will leverage
`hash::combine` to combine the hash values of aggregate fields. Manual
implementations which only want to hash a select number of fields would look
like:

```rust
impl Hash for MyType {
    fn hash(&self) -> usize {
        // ignore field2
        (&self.field1, &self.field3).hash()
    }
}
```

A possible implementation of combine can be found [in the boost source
code][boost-combine].

[boost-combine]: https://github.com/boostorg/functional/blob/master/include/boost/functional/hash/hash.hpp#L209-L213

## `HashMap` and DoS protection

Currently one of the features of the standard library's `HashMap` implementation
is that it by default provides DoS protection through two measures:

1. A strong hashing algorithm, SipHash 2-4, is used which is fairly difficult to
   find collisions with.
2. The SipHash algorithm is randomly seeded for each instance of `HashMap`. The
   algorithm is seeded with a 128-bit key.

These two measures ensure that each `HashMap` is randomly ordered, even if the
same keys are inserted in the same order. As a result, it is quite difficult to
mount a DoS attack against a `HashMap` as it is difficult to predict what
collisions will happen.

The `Hash` trait proposed above, however, does not allow SipHash to be
implemented generally any more. For example `#[derive(Hash)]` will no longer
leverage SipHash. Additionally, there is no input of state into the `hash`
function, so there is no random state per-`HashMap` to generate different hashes
with.

Denial of service attacks against hash maps are no new phenomenon, they are
[well](http://www.ocert.org/advisories/ocert-2011-003.html)
[known](http://lwn.net/Articles/474912/)
and have been reported in
[Python](http://bugs.python.org/issue13703),
[Ruby](https://www.ruby-lang.org/en/news/2011/12/28/denial-of-service-attack-was-found-for-rubys-hash-algorithm-cve-2011-4815/)
([other ruby](https://www.ruby-lang.org/en/news/2012/11/09/ruby19-hashdos-cve-2012-5371/)),
[Perl](http://blog.booking.com/hardening-perls-hash-function.html),
and many other languages/frameworks. Rust has taken a fairly proactive step from
the start by using a strong and randomly seeded algorithm since `HashMap`'s
inception.

In general the standard library does not provide many security-related
guarantees beyond memory safety. For example the new `Read::read_to_end`
function passes a safe buffer of uninitialized data to implementations of
`read` using various techniques to prevent memory safety issues. A DoS attack
against a hash map is such a common and well known exploit, however, that this
RFC considers it critical to consider the design of `Hash` and its relationship
with `HashMap`.

## Mitigation of DoS attacks

Other languages have mitigated DoS attacks via a few measures:

* [C++ specifies][cpp-hash] that the return value of `hash` is not guaranteed to
  be stable across program executions, allowing for a global salt to be mixed
  into hashes calculated.
* [Ruby has a global seed][ruby-seed] which is randomly initialized on startup
  and is used when hashing blocks of memory (e.g. strings).
* PHP and Tomcat have added limits to the maximum amount of keys allowed from a
  POST HTTP request (to limit the size of auto-generated maps). This strategy is
  not necessarily applicable to the standard library.

[cpp-hash]: http://en.cppreference.com/w/cpp/utility/hash
[ruby-seed]: https://github.com/ruby/ruby/blob/193ad64359b8ebcd77a2cba50a62d64311e26b22/random.c#L1248-L1251

It [has been claimed](http://bugs.python.org/issue13703#msg150558), however,
that a global seed may only mitigate some of the simplest attacks. The primary
downside is that a long-running process may leak the "global seed" through some
other form which could compromise maps in that specific process.

One possible route to mitigating these attacks with the `Hash` trait above could
be:

1. All primitives (integers, etc) are `combine`d with a global random seed which
   is initialized on first use.
2. Strings will continue to use SipHash as the default algorithm and the
   initialization keys will be randomly initialized on first use.

Given the information available about other DoS mitigations in hash maps for
other languages, however, it is not clear that this will provide the same level
of DoS protection that is available today. For example [@DaGenix explains
well](https://github.com/rust-lang/rfcs/pull/823#issuecomment-74013800) that we
may not be able to provide any form of DoS protection guarantee at all.

## Alternative Drawbacks

* One of the primary drawbacks to the proposed `Hash` trait is that it is now
  not possible to select an algorithm that a type should be hashed with. Instead
  each type's definition of hashing can only be altered through the use of a
  newtype wrapper.

* Today most Rust types can be hashed using a byte-oriented algorithm, so any
  number of these algorithms (e.g. SipHash, Fnv hashing) can be used. With this
  new `Hash` definition they are not easily accessible.

* Due to the lack of input state to hashing, the `HashMap` type can no longer
  randomly seed each individual instance but may at best have one global seed.
  This consequently elevates the risk of a DoS attack on a `HashMap` instance.

* The method of combining hashes together is not proven among other languages
  and is not guaranteed to provide the guarantees we want. This departure from
  the may have unknown consequences.

# Unresolved questions

* To what degree should `HashMap` attempt to prevent DoS attacks? Is it the
  responsibility of the standard library to do so or should this be provided as
  an external crate on crates.io?
