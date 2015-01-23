- Start Date: 2014-12-07
- RFC PR: [rust-lang/rfcs#517](https://github.com/rust-lang/rfcs/pull/517)
- Rust Issue: [rust-lang/rust#21070](https://github.com/rust-lang/rust/issues/21070)

# Summary
[Summary]: #summary

This RFC proposes a significant redesign of the `std::io` and `std::os` modules
in preparation for API stabilization. The specific problems addressed by the
redesign are given in the [Problems] section below, and the key ideas of the
design are given in [Vision for IO].

# Note about RFC structure

This RFC was originally posted as a single monolithic file, which made
it difficult to discuss different parts separately.

It has now been split into a skeleton that covers (1) the problem
statement, (2) the overall vision and organization, and (3) the
`std::os` module.

Other parts of the RFC are marked with `(stub)` and will be filed as
follow-up PRs against this RFC.

# Table of contents
[Table of contents]: #table-of-contents
* [Summary]
* [Table of contents]
* [Problems]
    * [Atomicity and the `Reader`/`Writer` traits]
    * [Timeouts]
    * [Posix and libuv bias]
    * [Unicode]
    * [stdio]
    * [Overly high-level abstractions]
    * [The error chaining pattern]
* [Detailed design]
    * [Vision for IO]
        * [Goals]
        * [Design principles]
            * [What cross-platform means]
            * [Relation to the system-level APIs]
            * [Platform-specific opt-in]
        * [Proposed organization]
    * [Revising `Reader` and `Writer`]
        * [Read]
        * [Write]
    * [String handling] (stub)
    * [Deadlines] (stub)
    * [Splitting streams and cancellation] (stub)
    * [Modules]
        * [core::io]
            * [Adapters]
            * [Free functions]
            * [Void]
            * [Seeking]
            * [Buffering]
            * [Cursor]
        * [The std::io facade]
            * [Errors]
            * [Channel adapters]
            * [stdin, stdout, stderr]
        * [std::env] (stub)
        * [std::fs] (stub)
        * [std::net] (stub)
        * [std::process] (stub)
        * [std::os]
    * [Odds and ends]
        * [The io prelude]
* [Drawbacks]
* [Alternatives]
* [Unresolved questions]

# Problems
[Problems]: #problems

The `io` and `os` modules are the last large API surfaces of `std` that need to
be stabilized. While the basic functionality offered in these modules is
*largely* traditional, many problems with the APIs have emerged over time. The
RFC discusses the most significant problems below.

This section only covers specific problems with the current library; see
[Vision for IO] for a higher-level view.  section.

## Atomicity and the `Reader`/`Writer` traits
[Atomicity and the `Reader`/`Writer` traits]: #atomicity-and-the-readerwriter-traits

One of the most pressing -- but also most subtle -- problems with `std::io` is
the lack of *atomicity* in its `Reader` and `Writer` traits.

For example, the `Reader` trait offers a `read_to_end` method:

```rust
fn read_to_end(&mut self) -> IoResult<Vec<u8>>
```

Executing this method may involve many calls to the underlying `read`
method. And it is possible that the first several calls succeed, and then a call
returns an `Err` -- which, like `TimedOut`, could represent a transient
problem. Unfortunately, given the above signature, there is no choice but to
simply _throw this data away_.

The `Writer` trait suffers from a more fundamental problem, since its primary
method, `write`, may actually involve several calls to the underlying system --
and if a failure occurs, there is no indication of how much was written.

Existing blocking APIs all have to deal with this problem, and Rust can and
should follow the existing tradition here. See [Revising `Reader` and `Writer`] for the proposed
solution.

## Timeouts
[Timeouts]: #timeouts

The `std::io` module supports "timeouts" on virtually all IO objects via a
`set_timeout` method. In this design, every IO object (file, socket, etc.) has
an optional timeout associated with it, and `set_timeout` mutates the associated
timeout. All subsequent blocking operations are implicitly subject to this timeout.

This API choice suffers from two problems, one cosmetic and the other deeper:

* The "timeout" is
  [actually a *deadline*](https://github.com/rust-lang/rust/issues/15802) and
  should be named accordingly.

* The stateful API has poor composability: when passing a mutable reference of
  an IO object to another function, it's possible that the deadline has been
  changed. In other words, users of the API can easily interfere with each other
  by accident.

See [Deadlines] for the proposed solution.

## Posix and libuv bias
[Posix and libuv bias]: #posix-and-libuv-bias

The current `io` and `os` modules were originally designed when `librustuv` was
providing IO support, and to some extent they reflect the capabilities and
conventions of `libuv` -- which in turn are loosely based on Posix.

As such, the modules are not always ideal from a cross-platform standpoint, both
in terms of forcing Windows programmings into a Posix mold, and also of offering
APIs that are not actually usable on all platforms.

The modules have historically also provided *no* platform-specific APIs.

Part of the goal of this RFC is to set out a clear and extensible story for both
cross-platform and platform-specific APIs in `std`. See [Design principles] for
the details.

## Unicode
[Unicode]: #unicode

Rust has followed the [utf8 everywhere](http://utf8everywhere.org/) approach to
its strings. However, at the borders to platform APIs, it is revealed that the
world is not, in fact, UTF-8 (or even Unicode) everywhere.

Currently our story for platform APIs is that we either assume they can take or
return Unicode strings (suitably encoded) or an uninterpreted byte
sequence. Sadly, this approach does *not* actually cover all platform needs, and
is also not highly ergonomic as presently implemented. (Consider `os::getev`
which introduces replacement characters (!) versus `os::getenv_as_bytes` which
yields a `Vec<u8>`; neither is ideal.)

This topic was covered in some detail in the
[Path Reform RFC](https://github.com/rust-lang/rfcs/pull/474), but this RFC
gives a more general account in [String handling].

## `stdio`
[stdio]: #stdio

The `stdio` module provides access to readers/writers for `stdin`, `stdout` and
`stderr`, which is essential functionality. However, it *also* provides a means
of changing e.g. "stdout" -- but there is no connection between these two! In
particular, `set_stdout` affects only the writer that `println!` and friends
use, while `set_stderr` affects `panic!`.

This module needs to be clarified. See [The std::io facade] and
[Functionality moved elsewhere] for the detailed design.

## Overly high-level abstractions
[Overly high-level abstractions]: #overly-high-level-abstractions

There are a few places where `io` provides high-level abstractions over system
services without also providing more direct access to the service as-is. For example:

* The `Writer` trait's `write` method -- a cornerstone of IO -- actually
  corresponds to an unbounded number of invocations of writes to the underlying
  IO object. This RFC changes `write` to follow more standard, lower-level
  practice; see [Revising `Reader` and `Writer`].

* Objects like `TcpStream` are `Clone`, which involves a fair amount of
  supporting infrastructure. This RFC tackles the problems that `Clone` was
  trying to solve more directly; see [Splitting streams and cancellation].

The motivation for going lower-level is described in [Design principles] below.

## The error chaining pattern
[The error chaining pattern]: #the-error-chaining-pattern

The `std::io` module is somewhat unusual in that most of the functionality it
proves are used through a few key traits (like `Reader`) and these traits are in
turn "lifted" over `IoResult`:

```rust
impl<R: Reader> Reader for IoResult<R> { ... }
```

This lifting and others makes it possible to chain IO operations that might
produce errors, without any explicit mention of error handling:

```rust
File::open(some_path).read_to_end()
                      ^~~~~~~~~~~ can produce an error
      ^~~~ can produce an error
```

The result of such a chain is either `Ok` of the outcome, or `Err` of the first
error.

While this pattern is highly ergonomic, it does not fit particularly well into
our evolving error story
([interoperation](https://github.com/rust-lang/rfcs/pull/201) or
[try blocks](https://github.com/rust-lang/rfcs/pull/243)), and it is the only
module in `std` to follow this pattern.

Eventually, we would like to write

```rust
File::open(some_path)?.read_to_end()
```

to take advantage of the `FromError` infrastructure, hook into error handling
control flow, and to provide good chaining ergonomics throughout *all* Rust APIs
-- all while keeping this handling a bit more explicit via the `?`
operator. (See https://github.com/rust-lang/rfcs/pull/243 for the rough direction).

In the meantime, this RFC proposes to phase out the use of impls for
`IoResult`. This will require use of `try!` for the time being.

(Note: this may put some additional pressure on at least landing the basic use
of `?` instead of today's `try!` before 1.0 final.)

# Detailed design
[Detailed design]: #detailed-design

There's a lot of material here, so the RFC starts with high-level goals,
principles, and organization, and then works its way through the various modules
involved.

## Vision for IO
[Vision for IO]: #vision-for-io

Rust's IO story had undergone significant evolution, starting from a
`libuv`-style pure green-threaded model to a dual green/native model and now to
a [pure native model](https://github.com/rust-lang/rfcs/pull/230). Given that
history, it's worthwhile to set out explicitly what is, and is not, in scope for
`std::io`

### Goals
[Goals]: #goals

For Rust 1.0, the aim is to:

* Provide a *blocking* API based directly on the services provided by the native
  OS for native threads.

  These APIs should cover the basics (files, basic networking, basic process
  management, etc) and suffice to write servers following the classic Apache
  thread-per-connection model. They should impose essentially zero cost over the
  underlying OS services; the core APIs should map down to a single syscall
  unless more are needed for cross-platform compatibility.

* Provide basic blocking abstractions and building blocks (various stream and
  buffer types and adapters) based on traditional blocking IO models but adapted
  to fit well within Rust.

* Provide hooks for integrating with low-level and/or platform-specific APIs.

* Ensure reasonable forwards-compatibility with future async IO models.

It is explicitly *not* a goal at this time to support asynchronous programming
models or nonblocking IO, nor is it a goal for the blocking APIs to eventually
be used in a nonblocking "mode" or style.

Rather, the hope is that the basic abstractions of files, paths, sockets, and so
on will eventually be usable directly within an async IO programing model and/or
with nonblocking APIs. This is the case for most existing languages, which offer
multiple interoperating IO models.

The *long term* intent is certainly to support async IO in some form,
but doing so will require new research and experimentation.

### Design principles
[Design principles]: #design-principles

Now that the scope has been clarified, it's important to lay out some broad
principles for the `io` and `os` modules. Many of these principles are already
being followed to some extent, but this RFC makes them more explicit and applies
them more uniformly.

#### What cross-platform means
[What cross-platform means]: #what-cross-platform-means

Historically, Rust's `std` has always been "cross-platform", but as discussed in
[Posix and libuv bias] this hasn't always played out perfectly. The proposed
policy is below. **With this policies, the APIs should largely feel like part of
"Rust" rather than part of any legacy, and they should enable truly portable
code**.

Except for an explicit opt-in (see [Platform-specific opt-in] below), all APIs
in `std` should be cross-platform:

* The APIs should **only expose a service or a configuration if it is supported on
  all platforms**, and if the semantics on those platforms is or can be made
  loosely equivalent. (The latter requires exercising some
  judgment). Platform-specific functionality can be handled separately
  ([Platform-specific opt-in]) and interoperate with normal `std` abstractions.

  This policy rules out functions like `chown` which have a clear meaning on
  Unix and no clear interpretation on Windows; the ownership and permissions
  models are *very* different.

* The APIs should **follow Rust's conventions**, including their naming, which
  should be platform-neutral.

  This policy rules out names like `fstat` that are the legacy of a particular
  platform family.

* The APIs should **never directly expose the representation** of underlying
  platform types, even if they happen to coincide on the currently-supported
  platforms. Cross-platform types in `std` should be newtyped.

  This policy rules out exposing e.g. error numbers directly as an integer type.

The next subsection gives detail on what these APIs should look like in relation
to system services.

#### Relation to the system-level APIs
[Relation to the system-level APIs]: #relation-to-the-system-level-apis

How should Rust APIs map into system services? This question breaks down along
several axes which are in tension with one another:

* **Guarantees**. The APIs provided in the mainline `io` modules should be
  predominantly safe, aside from the occasional `unsafe` function. In
  particular, the representation should be sufficiently hidden that most use
  cases are safe by construction. Beyond memory safety, though, the APIs should
  strive to provide a clear multithreaded semantics (using the `Send`/`Sync`
  kinds), and should use Rust's type system to rule out various kinds of bugs
  when it is reasonably ergonomic to do so (following the usual Rust
  conventions).

* **Ergonomics**. The APIs should present a Rust view of things, making use of
  the trait system, newtypes, and so on to make system services fit well with
  the rest of Rust.

* **Abstraction/cost**. On the other hand, the abstractions introduced in `std`
  must not induce significant costs over the system services -- or at least,
  there must be a way to safely access the services directly without incurring
  this penalty. When useful abstractions would impose an extra cost, they must
  be pay-as-you-go.

Putting the above bullets together, **the abstractions must be safe, and they
should be as high-level as possible without imposing a tax**.

* **Coverage**. Finally, the `std` APIs should over time strive for full
  coverage of non-niche, cross-platform capabilities.

#### Platform-specific opt-in
[Platform-specific opt-in]: #platform-specific-opt-in

Rust is a systems language, and as such it should expose seamless, no/low-cost
access to system services. In many cases, however, this cannot be done in a
cross-platform way, either because a given service is only available on some
platforms, or because providing a cross-platform abstraction over it would be
costly.

This RFC proposes *platform-specific opt-in*: submodules of `os` that are named
by platform, and made available via `#[cfg]` switches. For example, `os::unix`
can provide APIs only available on Unix systems, and `os::linux` can drill
further down into Linux-only APIs. (You could even imagine subdividing by OS
versions.) This is "opt-in" in the sense that, like the `unsafe` keyword, it is
very easy to audit for potential platform-specificity: just search for
`os::anyplatform`. Moreover, by separating out subsets like `linux`, it's clear
exactly how specific the platform dependency is.

The APIs in these submodules are intended to have the same flavor as other `io`
APIs and should interoperate seamlessly with cross-platform types, but:

* They should be named according to the underlying system services when there is
  a close correspondence.

* They may reveal the underlying OS type if there is nothing to be gained by
  hiding it behind an abstraction.

For example, the `os::unix` module could provide a `stat` function that takes a
standard `Path` and yields a custom struct. More interestingly, `os::linux`
might include an `epoll` function that could operate *directly* on many `io`
types (e.g. various socket types), without any explicit conversion to a file
descriptor; that's what "seamless" means.

Each of the platform modules will offer a custom `prelude` submodule,
intended for glob import, that includes all of the extension traits
applied to standard IO objects.

The precise design of these modules is in the very early stages and will likely
remain `#[unstable]` for some time.

### Proposed organization
[Proposed organization]: #proposed-organization

The `io` module is currently the biggest in `std`, with an entire hierarchy
nested underneath; it mixes general abstractions/tools with specific IO objects.
The `os` module is currently a bit of a dumping ground for facilities that don't
fit into the `io` category.

This RFC proposes the revamp the organization by flattening out the hierarchy
and clarifying the role of each module:

```
std
  env           environment manipulation
  fs            file system
  io            core io abstractions/adapters
    prelude     the io prelude
  net           networking
  os
    unix        platform-specific APIs
    linux         ..
    windows       ..
  os_str        platform-sensitive string handling
  process       process management
```

In particular:

* The contents of `os` will largely move to `env`, a new module for
inspecting and updating the "environment" (including environment variables, CPU
counts, arguments to `main`, and so on).

* The `io` module will include things like `Reader` and `BufferedWriter` --
  cross-cutting abstractions that are needed throughout IO.

  The `prelude` submodule will export all of the traits and most of the types
  for IO-related APIs; a single glob import should suffice to set you up for
  working with IO. (Note: this goes hand-in-hand with *removing* the bits of
  `io` currently in the prelude, as
  [recently proposed](https://github.com/rust-lang/rfcs/pull/503).)

* The root `os` module is used purely to house the platform submodules discussed
  [above](#platform-specific-opt-in).

* The `os_str` module is part of the solution to the Unicode problem; see
  [String handling] below.

* The `process` module over time will grow to include querying/manipulating
  already-running processes, not just spawning them.

## Revising `Reader` and `Writer`
[Revising `Reader` and `Writer`]: #revising-reader-and-writer

The `Reader` and `Writer` traits are the backbone of IO, representing
the ability to (respectively) pull bytes from and push bytes to an IO
object. The core operations provided by these traits follows a very
long tradition for blocking IO, but they are still surprisingly subtle
-- and they need to be revised.

* **Atomicity and data loss**. As discussed
  [above](#atomicity-and-the-reader-writer-traits), the `Reader` and
  `Writer` traits currently expose methods that involve multiple
  actual reads or writes, and data is lost when an error occurs after
  some (but not all) operations have completed.

  The proposed strategy for `Reader` operations is to (1) separate out
  various deserialization methods into a distinct framework, (2)
  *never* have the internal `read` implementations loop on errors, (3)
  cut down on the number of non-atomic read operations and (4) move
  the remaining operations to a different trait.

  For writers, the main
  change is to make `write` only perform a single underlying write
  (returning the number of bytes written on success), and provide a
  separate `write_all` method.

* **Parsing/serialization**. The `Reader` and `Writer` traits
  currently provide a large number of default methods for
  (de)serialization of various integer types to bytes with a given
  endianness. Unfortunately, these operations pose atomicity problems
  as well (e.g., a read could fail after reading two of the bytes
  needed for a `u32` value).

  Rather than complicate the signatures of these methods, the
  (de)serialization infrastructure is removed entirely -- in favor of
  instead eventually introducing a much richer
  parsing/formatting/(de)serialization framework that works seamlessly
  with `Reader` and `Writer`.

  Such a framework is out of scope for this RFC, but the
  endian-sensitive functionality will be provided elsewhere
  (likely out of tree).

* **The error type**. The traits currently use `IoResult` in their
  return types, which ties them to `IoError` in particular. Besides
  being an unnecessary restriction, this type prevents `Reader` and
  `Writer` (and various adapters built on top of them) from moving to
  `libcore` -- `IoError` currently requires the `String` type.

  With associated types, there is essentially no downside in making
  the error type generic.

With those general points out of the way, let's look at the details.

### `Read`
[Read]: #read

The updated `Reader` trait (and its extension) is as follows:

```rust
trait Read {
    type Err; // new associated error type

    // unchanged except for error type
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, Err>;
}

// extension trait needed for object safety
trait ReadExt: Read {
    fn bytes(&mut self) -> Bytes<Self> { ... }
    fn chars<'r>(&'r mut self) -> Chars<'r, Self, Err> { ... }

    fn read_to_end(&mut self) -> Result<Vec<u8>, Err> { ... }
    fn read_to_string(&self) -> Result<String, Err> { ... }

    ... // more to come later in the RFC
}
impl<R: Read> ReadExt for R {}
```

Following the
[trait naming conventions](https://github.com/rust-lang/rfcs/pull/344),
the trait is renamed to `Read` reflecting the single method it
provides.

The `read` method should not involve internal looping (even over
errors like `EINTR`). It is intended to be atomic.

#### Removed methods

The proposed `Read` trait is much slimmer than today's `Reader`. The vast
majority of removed methods are parsing/deserialization, which were
discussed above.

The remaining methods (`read_exact`, `read_at_least`, `push`,
`push_at_least`) were removed for various reasons:

* `read_exact`, `read_at_least`: these are somewhat more obscure
  conveniences that are not particularly robust due to lack of
  atomicity.

* `push`, `push_at_least`: these are special-cases for working with
  `Vec`, which this RFC proposes to replace with a more general
  mechanism described next.

To provide some of this functionality in a more composition way,
extend `Vec<T>` with an unsafe method:

```rust
unsafe fn with_extra(&mut self, n: uint) -> &mut [T];
```

This method is equivalent to calling `reserve(n)` and then providing a
slice to the memory starting just after `len()` entries. Using this
method, clients of `Read` can easily recover the `push` method.

### `Write`
[Write]: #write

The `Writer` trait is cut down to even smaller size:

```rust
trait Write {
    type Err;
    fn write(&mut self, buf: &[u8]) -> Result<uint, Err>;
    fn flush(&mut self) -> Result<(), Err> { ... }
}

trait WriteExt {
    fn write_all(&mut self, buf: &[u8]) -> Result<(), Err> { ... };
    fn write_fmt(&mut self, fmt: &fmt::Arguments) -> Result<(), Err> { ... }
}
impl<W: Write> WriteExt for W {}
```

The biggest change here is to the semantics of `write`. Instead of
repeatedly writing to the underlying IO object until all of `buf` is
written, it attempts a *single* write and on success returns the
number of bytes written. This follows the long tradition of blocking
IO, and is a more fundamental building block than the looping write we
currently have.

For convenience, `write_all` recovers the behavior of today's `write`,
looping until either the entire buffer is written or an error
occurs. Like `read_to_end`, if an error occurs before the operation is
complete, the intermediate result (of how much has been written) is
discarded. To meaningfully recover from an intermediate error, code
should work with `write` directly.

The `write_fmt` method, like `write_all`, will loop until its entire
input is written or an error occurs.

The other methods include endian conversions (covered by
serialization) and a few conveniences like `write_str` for other basic
types. The latter, at least, is already uniformly (and extensibly)
covered via the `write!` macro. The other helpers, as with `Read`,
should migrate into a more general (de)serialization library.

## String handling
[String handling]: #string-handling

> To be added in a follow-up PR.

## Deadlines
[Deadlines]: #deadlines

> To be added in a follow-up PR.

## Splitting streams and cancellation
[Splitting streams and cancellation]: #splitting-streams-and-cancellation

> To be added in a follow-up PR.

## Modules
[Modules]: #modules

Now that we've covered the core principles and techniques used
throughout IO, we can go on to explore the modules in detail.

### `core::io`
[core::io]: #coreio

The `io` module is split into the parts that can live in `libcore`
(most of it) and the parts that are added in the `std::io`
facade. Being able to move components into `libcore` at all is made
possible through the use of
[associated error types](#revising-reader-and-writer) for `Read` and
`Write`.

#### Adapters
[Adapters]: #adapters

The current `std::io::util` module offers a number of `Reader` and
`Writer` "adapters". This RFC refactors the design to more closely
follow `std::iter`. Along the way, it generalizes the `by_ref` adapter:

```rust
trait ReadExt: Read {
    // ... eliding the methods already described above

    // Reify a borrowed reader as owned
    fn by_ref(&mut self) -> ByRef<Self> { ... }

    // Map all errors to another type via `FromError`
    fn map_err<E: FromError<Self::Err>>(self) -> MapErr<Self, E> { ... }

    // Read everything from `self`, then read from `next`
    fn chain<R: Read>(self, next: R) -> Chain<Self, R> { ... }

    // Adapt `self` to yield only the first `limit` bytes
    fn take(self, limit: u64) -> Take<Self> { ... }

    // Whenever reading from `self`, push the bytes read to `out`
    #[unstable] // uncertain semantics of errors "halfway through the operation"
    fn tee<W: Write>(self, out: W) -> Tee<Self, W> { ... }
}

trait WriteExt: Write {
    // ... eliding the methods already described above

    // Reify a borrowed writer as owned
    fn by_ref<'a>(&'a mut self) -> ByRef<'a, Self> { ... }

    // Map all errors to another type via `FromError`
    fn map_err<E: FromError<Self::Err>>(self) -> MapErr<Self, E> { ... }

    // Whenever bytes are written to `self`, write them to `other` as well
    #[unstable] // uncertain semantics of errors "halfway through the operation"
    fn broadcast<W: Write>(self, other: W) -> Broadcast<Self, W> { ... }
}

// An adaptor converting an `Iterator<u8>` to `Read`.
pub struct IterReader<T> { ... }
```

As with `std::iter`, these adapters are object unsafe and hence placed
in an extension trait with a blanket `impl`.

Note that the same `ByRef` type is used for both `Read` and `Write`
-- and this RFC proposes to use it for `std::iter` as well. The
insight is that there is no difference between the *type* used for
by-ref adapters in any of these cases; what changes is just which
trait defers through it. So, we propose to add the following to `core::borrow`:

```rust
pub struct ByRef<'a, Sized? T:'a> {
    pub inner: &'a mut T
}
```

which will allow `impl`s like the following in `core::io`:

```rust
impl<'a, W: Write> Write for ByRef<'a, W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> Result<uint, W::Err> { self.inner.write(buf) }

    #[inline]
    fn flush(&mut self) -> Result<(), W::Err> { self.inner.flush() }
}
```

#### Free functions
[Free functions]: #free-functions

The current `std::io::util` module also includes a number of primitive
readers and writers, as well as `copy`. These are updated as follows:

```rust
// A reader that yields no bytes
fn empty() -> Empty; // in theory just returns `impl Read`

impl Read for Empty { type Err = Void; ... }

// A reader that yields `byte` repeatedly (generalizes today's ZeroReader)
fn repeat(byte: u8) -> Repeat;

impl Read for Repeat { type Err = Void; ... }

// A writer that ignores the bytes written to it (/dev/null)
fn sink() -> Sink;

impl Write for Sink { type Err = Void; ... }

// Copies all data from a `Read` to a `Write`, returning the amount of data
// copied.
pub fn copy<E, R, W>(r: &mut R, w: &mut W) -> Result<u64, E> where
    R: Read<Err = E>,
    W: Write<Err = E>
```

Like `write_all`, the `copy` method will discard the amount of data already
written on any error and also discard any partially read data on a `write`
error. This method is intended to be a convenience and `write` should be used
directly if this is not desirable.

#### Void
[Void]: #void

A new concrete error type will be added in the standard library. A new module
`std::void` will be introduced with the following contents:

```rust
pub enum Void {}

impl<E: Error> FromError<Void> for E {
    fn from_error(v: Void) -> E {
        match v {}
    }
}
```

Applications for an uninhabited enum have come up from time-to-time, and some of
the core I/O adapters represent a fairly concrete use case motivating its
existence. By using an associated `Err` type of `Void`, each I/O object is
indicating that it *can never fail*. This allows the types themselves to be more
optimized in the future as well as enabling interoperation with many other error
types via the `map_err` adaptor.

Some possible future optimizations include:

* `Result<T, Void>` could be represented in memory exactly as `T` (no
  discriminant).
* The `unused_must_use` lint could understand that `Result<T, Void>` does not
  need to be warned about.

This RFC does not propose implementing these modifications at this time,
however.

#### Seeking
[Seeking]: #seeking

The seeking infrastructure is largely the same as today's, except that
`tell` is removed and the `seek` signature is refactored with more precise
types:

```rust
pub trait Seek {
    type Err;
    // returns the new position after seeking
    fn seek(&mut self, pos: SeekPos) -> Result<u64, Err>;
}

pub enum SeekPos {
    FromStart(u64),
    FromEnd(i64),
    FromCur(i64),
}
```

The old `tell` function can be regained via `seek(SeekPos::FromCur(0))`.

#### Buffering
[Buffering]: #buffering

The current `Buffer` trait will be renamed to `BufferedRead` for
clarity (and to open the door to `BufferedWrite` at some later
point):

```rust
pub trait BufferedRead: Read {
    fn fill_buf(&mut self) -> Result<&[u8], Self::Err>;
    fn consume(&mut self, amt: uint);
}

pub trait BufferedReadExt: BufferedRead {
    fn read_until(&mut self, byte: u8) -> ReadUntil { ... }
    fn lines(&mut self) -> Lines<Self, Self::Err> { ... };
}
```

where `ReadUntil: Iterator<Result<Vec<u8>, Self::Err>>`.  In addition,
`read_line` is removed in favor of the `lines` iterator, and
`read_char` is removed in favor of the `chars` iterator (now on
`ReadExt`). As with all nonatomic convenience methods, these methods
do not offer a means of recovering from a transient errors; one should
use the lower-level methods in such cases.

The `BufferedReader`, `BufferedWriter` and `BufferedStream` types stay
essentially as they are today, except that for streams and writers the
`into_inner` method yields any errors encountered when flushing,
together with the remaining data:

```rust
// If flushing fails, you get the unflushed data back
fn into_inner(self) -> Result<W, (Vec<u8>, W::Err)>;
```

#### `Cursor`
[Cursor]: #cursor

Many applications want to view in-memory data as either an implementor of `Read`
or `Write`. This is often useful when composing streams or creating test cases.
This functionality primarily comes from the following implementations:

```rust
impl<'a> Read for &'a [u8] { type Err = Void; ... }
impl<'a> Write for &'a mut [u8] { type Err = Void; ... }
impl Write for Vec<u8> { type Err = Void; ... }
```

While efficient, none of these implementations support seeking (via an
implementation of the `Seek` trait). The implementations of `Read` and `Write`
for these types is not quite as efficient when `Seek` needs to be used, so the
`Seek`-ability will be opted-in to with a new `Cursor` structure with the
following API:

```rust
pub struct Cursor<T> {
    pos: u64,
    inner: T,
}
impl<T> Cursor<T> {
    pub fn new(inner: T) -> Cursor<T>;
    pub fn into_inner(self) -> T;
    pub fn get_ref(&self) -> &T;
}

// Error indicating that a negative offset was seeked to.
pub struct NegativeOffset;

impl Seek for Cursor<Vec<u8>> { type Err = NegativeOffset; ... }
impl<'a> Seek for Cursor<&'a [u8]> { type Err = NegativeOffset; ... }
impl<'a> Seek for Cursor<&'a mut [u8]> { type Err = NegativeOffset; ... }

impl Read for Cursor<Vec<u8>> { type Err = Void; ... }
impl<'a> Read for Cursor<&'a [u8]> { type Err = Void; ... }
impl<'a> Read for Cursor<&'a mut [u8]> { type Err = Void; ... }

impl BufferedRead for Cursor<Vec<u8>> { type Err = Void; ... }
impl<'a> BufferedRead for Cursor<&'a [u8]> { type Err = Void; ... }
impl<'a> BufferedRead for Cursor<&'a mut [u8]> { type Err = Void; ... }

impl<'a> Write for Cursor<&'a mut [u8]> { type Err = Void; ... }
impl Write for Cursor<Vec<u8>> { type Err = Void; ... }
```

A sample implementation can be found in [a gist][cursor-impl]. Using one
`Cursor` structure allows to emphasize that the only ability added is an
implementation of `Seek` while still allowing all possible I/O operations for
various types of buffers.

[cursor-impl]: https://gist.github.com/alexcrichton/8224f57ed029929447bd

It is not currently proposed to unify these implementations via a trait. For
example a `Cursor<Rc<[u8]>>` is a reasonable instance to have, but it will not
have an implementation listed in the standard library to start out. It is
considered a backwards-compatible addition to unify these various `impl` blocks
with a trait.

The following types will be removed from the standard library and replaced as
follows:

* `MemReader` -> `Cursor<Vec<u8>>`
* `MemWriter` -> `Cursor<Vec<u8>>`
* `BufReader` -> `Cursor<&[u8]>` or `Cursor<&mut [u8]>`
* `BufWriter` -> `Cursor<&mut [u8]>`

### The `std::io` facade
[The std::io facade]: #the-stdio-facade

The `std::io` module will largely be a facade over `core::io`, but it
will add some functionality that can live only in `std`.

#### `Errors`
[Errors]: #error

The `IoError` type will be renamed to `std::io::Error`, following our
[non-prefixing convention](https://github.com/rust-lang/rfcs/pull/356).
It will remain largely as it is today, but its fields will be made
private. It may eventually grow a field to track the underlying OS
error code.

The `std::io::IoErrorKind` type will become `std::io::ErrorKind`, and
`ShortWrite` will be dropped (it is no longer needed with the new
`Write` semantics), which should decrease its footprint. The
`OtherIoError` variant will become `Other` now that `enum`s are
namespaced. Other variants may be added over time, such as `Interrupted`,
as more errors are classified from the system.

The `EndOfFile` variant will be removed in favor of returning `Ok(0)`
from `read` on end of file (or `write` on an empty slice for example). This
approach clarifies the meaning of the return value of `read`, matches Posix
APIs, and makes it easier to use `try!` in the case that a "real" error should
be bubbled out. (The main downside is that higher-level operations that might
use `Result<T, IoError>` with some `T != usize` may need to wrap `IoError` in a
further enum if they wish to forward unexpected EOF.)

#### Channel adapters
[Channel adapters]: #channel-adapters

The `ChanReader` and `ChanWriter` adapters will be left as they are today, and
they will remain `#[unstable]`. The channel adapters currently suffer from a few
problems today, some of which are inherent to the design:

* Construction is somewhat unergonomic. First a `mpsc` channel pair must be
  created and then each half of the reader/writer needs to be created.
* Each call to `write` involves moving memory onto the heap to be sent, which
  isn't necessarily efficient.
* The design of `std::sync::mpsc` allows for growing more channels in the
  future, but it's unclear if we'll want to continue to provide a reader/writer
  adapter for each channel we add to `std::sync`.

These types generally feel as if they're from a different era of Rust (which
they are!) and may take some time to fit into the current standard library. They
can be reconsidered for stabilization after the dust settles from the I/O
redesign as well as the recent `std::sync` redesign. At this time, however, this
RFC recommends they remain unstable.

#### `stdin`, `stdout`, `stderr`
[stdin, stdout, stderr]: #stdin-stdout-stderr

Finally, `std::io` will provide a `stdin` reader and `stdout` and
`stderr` writers. These will largely work as they do today, except
that we will hew more closely to the traditional setup:

* `stderr` will be unbuffered and `stderr_raw` will therefore be dropped.
* `stdout` will be line-buffered for TTY, fully buffered otherwise.
* most TTY functionality in `StdReader` and `StdWriter` will be moved
   to `os::unix`, since it's not yet implemented on Windows.
* `stdout_raw` and `stderr_raw` will be removed.

### `std::env`
[std::env]: #stdenv

> To be added in a follow-up PR.

### `std::fs`
[std::fs]: #stdfs

> To be added in a follow-up PR.

### `std::net`
[std::net]: #stdnet

> To be added in a follow-up PR.

### `std::process`
[std::process]: #stdprocess

> To be added in a follow-up PR.

### `std::os`
[std::os]: #stdos

Initially, this module will be empty except for the platform-specific
`unix` and `windows` modules. It is expected to grow additional, more
specific platform submodules (like `linux`, `macos`) over time.

## Odds and ends
[Odds and ends]: #odds-and-ends

> To be expanded in a follow-up PR.

### The `io` prelude
[The io prelude]: #the-io-prelude

The `prelude` submodule will contain most of the traits, types, and
modules discussed in this RFC; it is meant to provide maximal
convenience when working with IO of any kind. The exact contents of
the module are left as an open question.

# Drawbacks
[Drawbacks]: #drawbacks

This RFC is largely about cleanup, normalization, and stabilization of
our IO libraries -- work that needs to be done, but that also
represents nontrivial churn.

However, the actual implementation work involved is estimated to be
reasonably contained, since all of the functionality is already in
place in some form (including `os_str`, due to @SimonSapin's
[WTF-8 implementation](https://github.com/SimonSapin/rust-wtf8)).

# Alternatives
[Alternatives]: #alternatives

The main alternative design would be to continue staying with the
Posix tradition in terms of naming and functionality (for which there
is precedent in some other languages). However, Rust is already
well-known for its strong cross-platform compatibility in `std`, and
making the library more Windows-friendly will only increase its appeal.

More radically different designs (in terms of different design
principles or visions) are outside the scope of this RFC.

# Unresolved questions
[Unresolved questions]: #unresolved-questions

> To be expanded in a follow-up PR.
