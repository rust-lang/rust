- Start Date: 2014-12-07
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[Summary]: #summary

This RFC proposes a significant redesign of the `std::io` and `std::os` modules
in preparation for API stabilization. The specific problems addressed by the
redesign are given in the [Problems] section below, and the key ideas of the
design are given in [Vision for IO].

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
        * [Nonatomic results]
        * [Reader]
        * [Writer]
    * [String handling]
        * [Key observations]
        * [The design: `os_str`]
        * [The future]
    * [Deadlines]
        * [Timeouts versus deadlines]
    * [Splitting streams and cancellation]
    * [Modules]
        * [core::io]
            * [Adapters]
            * [Seeking]
            * [Buffering]
            * [MemReader and MemWriter]
        * [The std::io facade]
            * [Errors]
            * [Channel adapters]
            * [stdin, stdout, stderr]
        * [std::env]
        * [std::fs]
            * [Free functions]
            * [Files]
            * [File kinds]
            * [File permissions]
        * [std::net]
            * [TCP]
            * [UDP]
            * [Addresses]
        * [std::process]
            * [Command]
            * [Child]
        * [std::os]
    * [Odds and ends]
        * [The io prelude]
        * [Functionality moved elsewhere]
        * [Functionality removed outright]
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

The *long term* intent is certainly to support async IO in some form, which is
needed for some kinds of high-performance servers among other things. But doing
so will require new research and experimentation.

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

  The proposed strategy for `Reader` operations is to return the
  already-read data together with an error. For writers, the main
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

  Such a framework is out of scope for this RFC, however, so the
  endian-sensitive functionality will likely be provided elsewhere
  (likely out of tree).

* **The error type**. The traits currently use `IoResult` in their
  return types, which ties them to `IoError` in particular. Besides
  being an unnecessary restriction, this type prevents `Reader` and
  `Writer` (and various adapters built on top of them) from moving to
  `libcore` -- `IoError` currently requires the `String` type.

  With associated types, there is essentially no downside in making
  the error type generic.

With those general points out of the way, let's look at the details.

### Nonatomic results
[Nonatomic results]: #nonatomic-results

To clarity dealing with nonatomic operations and improve their
ergonomics, we introduce some new types into `std::error`:

```rust
// The progress so far (T) paired with an err (Err)
struct PartialResult<T, Err>(T, Err);

// An operation that may fail after having made some progress:
// - S is what's produced on complete success,
// - T is what's produced if an operation fails part of the way through
type NonatomicResult<S, T, Err> = Result<S, PartialResult<T, Err>>;

// Ergonomically throw out the partial result
impl<T, Err> FromError<PartialResult<T, Err> for Err { ... }
```

The `NonatomicResult` type (which could use a shorter name)
encapsulates the common pattern of operations that may fail after
having made some progress. The `PartialResult` type then returns the
progress that was made along with the error, but with a `FromError`
implementation that makes it trivial to throw out the partial result
if desired.

### `Reader`
[Reader]: #reader

The updated `Reader` trait (and its extension) is as follows:

```rust
trait Reader {
    type Err; // new associated error type

    // unchanged except for error type
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, Err>;

    // these all return partial results on error
    fn read_to_end(&mut self) -> NonatomicResult<Vec<u8>, Vec<u8>, Err> { ... }
    fn read_to_string(&self) -> NonatomicResult<String, Vec<u8>, Err> { ... }
    fn read_at_least(&mut self, min: uint,  buf: &mut [u8]) -> NonatomicResult<uint, uint, Err>  { ... }
}

// extension trait needed for object safety
trait ReaderExt: Reader {
    fn bytes(&mut self) -> Bytes<Self> { ... }
    fn chars<'r>(&'r mut self) -> Chars<'r, Self, Err> { ... }

    ... // more to come later in the RFC
}
impl<R: Reader> ReaderExt for R {}
```

#### Removed methods

The proposed `Reader` trait is much slimmer than today's. The vast
majority of removed methods are parsing/deserialization, which were
discussed above.

The remaining methods (`read_exact`, `push`, `push_at_least`) were
removed largely because they are *not memory safe*: they involve
extending a vector's capacity, and then *passing in the resulting
uninitialized memory* to the `read` method, which is not marked
`unsafe`! Thus the current design can lead to undefined behavior in
safe code.

The solution is to instead extend `Vec<T>` with a useful unsafe method:

```rust
unsafe fn with_extra(&mut self, n: uint) -> &mut [T];
```

This method is equivalent to calling `reserve(n)` and then providing a
slice to the memory starting just after `len()` entries. Using this
method, clients of `Reader` can easily recover the above removed
methods, but they are explicitly marking the unsafety of doing so.

(Note: `read_to_end` is currently not memory safe for the same reason,
but is considered a very important convenience. Thus, we will continue
to provide it, but will zero the slice beforehand.)

### `Writer`
[Writer]: #writer

The `Writer` trait is cut down to even smaller size:

```rust
trait Writer {
    type Err;
    fn write(&mut self, buf: &[u8]) -> Result<uint, Err>;

    fn write_all(&mut self, buf: &[u8]) -> NonatomicResult<(), uint, Err> { ... };
    fn write_fmt(&mut self, fmt: &fmt::Arguments) -> Result<(), Err> { ... }
    fn flush(&mut self) -> Result<(), Err> { ... }
}
```

The biggest change here is to the semantics of `write`. Instead of
repeatedly writing to the underlying IO object until all of `buf` is
written, it attempts a *single* write and on success returns the
number of bytes written. This follows the long tradition of blocking
IO, and is a more fundamental building block than the looping write we
currently have.

For convenience, `write_all` recovers the behavior of today's `write`,
looping until either the entire buffer is written or an error
occurs. In the latter case, however, it now also yields the number of
bytes that had been written prior to the error.

The `write_fmt` method, like `write_all`, will loop until its entire
input is written or an error occurs. However, it does not return a
`NonatomicResult` because the number of bytes written cannot be
straightforwardly interpreted -- the actual byte sequence written is
determined by the formatting system.

The other methods include endian conversions (covered by
serialization) and a few conveniences like `write_str` for other basic
types. The latter, at least, is already uniformly (and extensibly)
covered via the `write!` macro. The other helpers, as with `Reader`,
should migrate into a more general (de)serialization library.

## String handling
[String handling]: #string-handling

The fundamental problem with Rust's full embrace of UTF-8 strings is that not
all strings taken or returned by system APIs are Unicode, let alone UTF-8
encoded.

In the past, `std` has assumed that all strings are *either* in some form of
Unicode (Windows), *or* are simply `u8` sequences (Unix). Unfortunately, this is
wrong, and the situation is more subtle:

* Unix platforms do indeed work with arbitrary `u8` sequences (without interior
  nulls) and today's platforms usually interpret them as UTF-8 when displayed.

* Windows, however, works with *arbitrary `u16` sequences* that are roughly
  interpreted at UTF-16, but may not actually be valid UTF-16 -- an "encoding"
  often call UCS-2; see http://justsolve.archiveteam.org/wiki/UCS-2 for a bit
  more detail.

What this means is that all of Rust's platforms go beyond Unicode, but they do
so in different and incompatible ways.

The current solution of providing both `str` and `[u8]` versions of
APIs is therefore problematic for multiple reasons. For one, **the
`[u8]` versions are not actually cross-platform** -- even today, they
panic on Windows when given non-UTF-8 data, a platform-specific
behavior. But they are also incomplete, because on Windows you should
be able to work directly with UCS-2 data.

### Key observations
[Key observations]: #key-observations

Fortunately, there is a solution that fits well with Rust's UTF-8 strings *and*
offers the possibility of platform-specific APIs.

**Observation 1**: it is possible to re-encode UCS-2 data in a way that is also
  compatible with UTF-8. This is the
  [WTF-8 encoding format](http://simonsapin.github.io/wtf-8/) proposed by Simon
  Sapin. This encoding has some remarkable properties:

* Valid UTF-8 data is valid WTF-8 data. When decoded to UCS-2, the result is
  exactly what would be produced by going straight from UTF-8 to UTF-16. In
  other words, making up some methods:

  ```rust
  my_ut8_data.to_wtf_8().to_ucs2().as_u16_slice() == my_utf8_data.to_utf16().as_16_slice()
  ```

* Valid UTF-16 data re-encoded as WTF-8 produces the corresponding UTF-8 data:

  ```rust
  my_utf16_data.to_wtf_8().as_bytes() == my_utf16_data.to_utf8().as_bytes()
  ```

These two properties mean that, when working with Unicode data, the WTF-8
encoding is highly compatible with both UTF-8 *and* UTF-16. In particular, the
conversion from a Rust string to a WTF-8 string is a no-op, and the conversion
in the other direction is just a validation.

**Observation 2**: all platforms can *consume* Unicode data (suitably
  re-encoded), and it's also possible to validate the data they produce as
  Unicode and extract it.

**Observation 3**: the non-Unicode spaces on various platforms are deeply
  incompatible: there is no standard way to port non-Unicode data from one to
  another. Therefore, the only cross-platform APIs are those that work entirely
  with Unicode.

### The design: `os_str`
[The design: `os_str`]: #the-design-os_str

The observations above lead to a somewhat radical new treatment of strings,
first proposed in the
[Path Reform RFC](https://github.com/rust-lang/rfcs/pull/474). This RFC proposes
to introduce new string and string slice types that (opaquely) represent
*platform-sensitive strings*, housed in the `std::os_str` module.

The `OsStrBuf` type is analogous to `String`, and `OsStr` is analogous to `str`.
Their backing implementation is platform-dependent, but they offer a
cross-platform API:

```rust
pub mod os_str {
    /// Owned OS strings
    struct OsStrBuf {
        inner: imp::Buf
    }
    /// Slices into OS strings
    struct OsStr {
        inner: imp::Slice
    }

    // Platform-specific implementation details:
    #[cfg(unix)]
    mod imp {
        type Buf = Vec<u8>;
        type Slice = [u8;
        ...
    }

    #[cfg(windows)]
    mod imp {
        type Buf = Wtf8Buf; // See https://github.com/SimonSapin/rust-wtf8
        type Slice = Wtf8;
        ...
    }

    impl OsStrBuf {
        pub fn from_string(String) -> OsStrBuf;
        pub fn from_str(&str) -> OsStrBuf;
        pub fn as_slice(&self) -> &OsStr;
        pub fn into_string(Self) -> Result<String, OsStrBuf>;
        pub fn into_string_lossy(Self) -> String;

        // and ultimately other functionality typically found on vectors,
        // but CRUCIALLY NOT as_bytes
    }

    impl Deref<OsStr> for OsStrBuf { ... }

    impl OsStr {
        pub fn from_str(value: &str) -> &OsStr;
        pub fn as_str(&self) -> Option<&str>;
        pub fn to_string_lossy(&self) -> CowString;

        // and ultimately other functionality typically found on slices,
        // but CRUCIALLY NOT as_bytes
    }

    trait IntoOsStrBuf {
        fn into_os_str_buf(self) -> OsStrBuf;
    }

    impl IntoOsStrBuf for OsStrBuf { ... }
    impl<'a> IntoOsStrBuf for &'a OsStr { ... }

    ...
}
```

These APIs make OS strings appear roughly as opaque vectors (you
cannot see the byte representation directly), and can always be
produced starting from Unicode data. They make it possible to collapse
functions like `getenv` and `getenv_as_bytes` into a single function
that produces an OS string, allowing the client to decide how (or
whether) to extract Unicode data. It will be possible to do things
like concatenate OS strings without ever going through Unicode.

It will also likely be possible to do things like search for Unicode
substrings. The exact details of the API are left open and are likely
to grow over time.

In addition to APIs like the above, there will also be
platform-specific ways of viewing or constructing OS strings that
reveals more about the space of possible values:

```rust
pub mod os {
    #[cfg(unix)]
    pub mod unix {
        trait OsStrBufExt {
            fn from_vec(Vec<u8>) -> Self;
            fn into_vec(Self) -> Vec<u8>;
        }

        impl OsStrBufExt for os_str::OsStrBuf { ... }

        trait OsStrExt {
            fn as_byte_slice(&self) -> &[u8];
            fn from_byte_slice(&[u8]) -> &Self;
        }

        impl OsStrExt for os_str::OsStr { ... }

        ...
    }

    #[cfg(windows)]
    pub mod windows{
        // The following extension traits provide a UCS-2 view of OS strings

        trait OsStrBufExt {
            fn from_wide_slice(&[u16]) -> Self;
        }

        impl OsStrBufExt for os_str::OsStrBuf { ... }

        trait OsStrExt {
            fn to_wide_vec(&self) -> Vec<u16>;
        }

        impl OsStrExt for os_str::OsStr { ... }

        ...
    }

    ...
}
```

By placing these APIs under `os`, using them requires a clear *opt in*
to platform-specific functionality.

### The future
[The future]: #the-future

Introducing an additional string type is a bit daunting, since many
existing APIs take and consume only standard Rust strings. Today's
solution demands that strings coming from the OS be assumed or turned
into Unicode, and the proposed API continues to allow that (with more
explicit and finer-grained control).

In the long run, however, robust applications are likely to work
opaquely with OS strings far beyond the boundary to the system to
avoid data loss and ensure maximal compatibility. If this situation
becomes common, it should be possible to introduce an abstraction over
various string types and generalize most functions that work with
`String`/`str` to instead work generically. This RFC does *not*
propose taking any such steps now -- but it's important that we *can*
do so later if Rust's standard strings turn out to not be sufficient
and OS strings become commonplace.

## Deadlines
[Deadlines]: #deadlines

Most blocking system operations can take a timeout or a deadline
(depending on the platform) for completion, and it's important that
Rusts IO APIs offer the same capability. This poses a bit of a
challenge, however, because adding variants to all of the blocking
APIs would significantly increase the API surface, while taking an
`Option` argument would decrease their ergonomics.

The current solution is to offer `set_timeout` methods on various IO
objects (a variant of a builder-style API), which allows configuration
to be done independently of the blocking operation being configured.

Unfortunately, as explained [above](#timeouts), this stateful approach
has poor composability, since users of an IO objects can accidentally
interfere with one another.

The proposed solution is to instead offer a `with_deadline` method
(correcting the terminology) that, rather than changing the state of
an object, creates a *wrapper* object with the given deadline.

```rust
struct Deadlined<T> {
    deadline: Duration,
    inner: T,
}

impl<T> Deadlined<T> {
    pub fn new(inner: T, deadline: Duration) -> Deadlined<T> {
        Deadlined { deadline: deadline, inner: inner }
    }

    pub fn deadline(&self) -> Duration {
        self.deadline
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl TcpStream {
    fn with_deadline(&mut self, deadline: Duration) -> Deadlined<&mut TcpStream> {
        Deadlined::new(self, deadline)
    }
}

impl<'a> Reader for Deadlined<&'a mut TcpStream> {
    type Err = IoError;
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, Err> {
        // read, using the specified deadline
    }
}

// And so on for other traits and concrete types
```

### Timeouts versus deadlines
[Timeouts versus deadlines]: #timeouts-versus-deadlines

This RFC is not going to delve deeply into the timeout versus deadline
debate, but the main motivation for using deadlines is for *compound*
operations like `write_all`. With a deadline-based approach, it's
possible to bound the total amount of time taken even though an
operation involves many system calls. Doing so with timeouts is
harder, since the timeout must change as the operations progress
(requiring re-checking the clock each time).

## Splitting streams and cancellation
[Splitting streams and cancellation]: #splitting-streams-and-cancellation

In the current design of `std::io`, types like `TcpStream` serve as
both readers and writers -- which means that, in order for one thread
to read from the stream and another to write, the stream must be
`Clone`.  A side benefit of cloneability is that one side of a stream
can be closed (via `close_read`, for example), which effectively
cancels any blocking operations in progress, allowing for a clean shutdown of
other threads.

While `Clone` addresses important use cases, it has some downsides. It
requires that the IO object internally use an `Arc`, and it means that
the implementation must be fully thread-safe. More generally, it
increases the distance from the underlying descriptors and adds
potentially costly functionality on the Rust side -- going against
this RFC's [Design principles] and potentially making forward
compatibility with async IO more difficult.

This RFC proposes to solve stream splitting and cancellation in a different way:

* To allow splitting up a stream, we will provide separable `Reader`
  and `Writer` components that can be extracted from an owned
  stream. That means that there is precisely one owner of each side
  (so thread safety is no longer an issue). An `Arc` is still needed
  to know when the entire stream should be dropped.

* To provide cancellation, we will make it possible to acquire a
  "cancellation token" for each side of a stream. This token can be
  freely cloned, and can be used to shutdown that side of the stream,
  cancelling any in-progress blocking operations. *But you pay for
  these tokens only if you use them*.

The details of this design will be given concretely in the section on
[std::net].

## Modules
[Modules]: #modules

Now that we've covered the core principles and techniques used
throughout IO, we can go on to explore the modules in detail.

### `core::io`
[core::io]: #coreio

The `io` module is split into a the parts that can live in `libcore`
(most of it) and the parts that are added in the `std::io`
facade. Being able to move components into `libcore` at all is made
possible through the use of
[associated error types](#revising-reader-and-writer) for `Reader` and
`Writer`.

#### Adapters
[Adapters]: #adapters

The current `std::io::util` module offers a number of `Reader` and
`Writer` "adapters". This RFC refactors the design to more closely
follow `std::iter`. Along the way, it generalizes the `by_ref` adapter:

```rust
trait ReaderExt: Reader {
    // already introduced above
    fn bytes(&mut self) -> Bytes<Self> { ... }
    fn chars<'r>(&'r mut self) -> Chars<'r, Self, Err> { ... }

    // Reify a borrowed reader as owned
    fn by_ref<'a>(&'a mut self) -> ByRef<'a, Self> { ... }

    // Read everything from `self`, then read from `next`
    fn chain<R: Reader>(self, next: R) -> Chain<Self, R> { ... }

    // Adapt `self` to yield only the first `limit` bytes
    fn take(self, limit: u64) -> Take<Self> { ... }

    // Whenever reading from `self`, push the bytes read to `out`
    fn tee<W: Writer>(self, out: W) -> Tee<Self, W> { ... }
}
impl<T: Reader> ReaderExt for T {}

trait WriterExt: Writer {
    // Reify a borrowed writer as owned
    fn by_ref<'a>(&'a mut self) -> ByRef<'a, Self> { ... }

    // Whenever bytes are written to `self`, write them to `other` as well
    fn carbon_copy<W: Writer>(self, other: W) -> CarbonCopy<Self, W> { ... }
}
impl<T: Writer> WriterExt for T {}

// An adaptor converting an `Iterator<u8>` to a `Reader`.
pub struct IterReader<T> { ... }
```

As with `std::iter`, these adapters are object unsafe an hence placed
in an extension trait with a blanket `impl`.

Note that the same `ByRef` type is used for both `Reader` and `Writer`
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
impl<'a, W: Writer> Writer for ByRef<'a, W> {
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
fn empty() -> Empty;

// A reader that yields `byte` repeatedly (generalizes today's ZeroReader)
fn repeat(byte: u8) -> Repeat;

// A writer that ignores the bytes written to it (/dev/null)
fn sink() -> Sink;

// Copies all data from a Reader to a Writer
pub fn copy<E, R, W>(r: &mut R, w: &mut W) -> NonatomicResult<(), uint, E> where
    R: Reader<Err = E>,
    W: Writer<Err = E>
```

#### Seeking
[Seeking]: #seeking

The seeking infrastructure is largely the same as today's, except that
`tell` is renamed to follow the RFC's design principles and the `seek`
signature is refactored with more precise types:

```rust
pub trait Seek {
    type Err;
    fn position(&self) -> Result<u64, Err>;
    fn seek(&mut self, pos: SeekPos) -> Result<(), Err>;
}

pub enum SeekPos {
    FromStart(u64),
    FromEnd(u64),
    FromCur(i64),
}
```

#### Buffering
[Buffering]: #buffering

The current `Buffer` trait will be renamed to `BufferedReader` for
clarity (and to open the door to `BufferedWriter` at some later
point):

```rust
pub trait BufferedReader: Reader {
    fn fill_buf(&mut self) -> Result<&[u8], Self::Err>;
    fn consume(&mut self, amt: uint);

    // This should perhaps yield an iterator
    fn read_until(&mut self, byte: u8) -> NonatomicResult<Vec<u8>, Vec<u8>, Self::Err> { ... }
}

pub trait BufferedReaderExt: BufferedReader {
    fn lines(&mut self) -> Lines<Self, Self::Err> { ... };
}
```

In addition, `read_line` is removed in favor of the `lines` iterator,
and `read_char` is removed in favor of the `chars` iterator (now on
`ReaderExt`). These iterators will be changed to yield
`NonatomicResult` values.

The `BufferedReader`, `BufferedWriter` and `BufferedStream` types stay
essentially as they are today, except that for streams and writers the
`into_inner` method yields any errors encountered when flushing,
together with the remaining data:

```rust
// If flushing fails, you get the unflushed data back
fn into_inner(self) -> NonatomicResult<W, Vec<u8>, W::Err>;
```

#### `MemReader` and `MemWriter`
[MemReader and MemWriter]: #memreader-and-memwriter

The various in-memory readers and writers available today will be
consolidated into just `MemReader` and `MemWriter`:

`MemReader` (like today's `BufReader`)
 - construct from `&[u8]`
 - implements `Seek`

`MemWriter`
 - construct freshly, or from a `Vec`
 - implements `Seek`

Both will allow decomposing into their inner parts, though the exact
details are left to the implementation.

The rationale for this design is that, if you want to read from a
`Vec`, it's easy enough to get a slice to read from instead; on the
other hand, it's rare to want to write into a mutable slice on the
stack, as opposed to an owned vector. So these two readers and writers
cover the vast majority of in-memory readers/writers for Rust.

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

The `IoErrorKind` type will become `std::io::ErrorKind`, and
`ShortWrite` will be dropped (it is no longer needed with the new
`Writer` semantics), which should decrease its footprint. The
`OtherIoError` variant will become `Other` now that `enum`s are
namespaced.

#### Channel adapters
[Channel adapters]: #channel-adapters

The `ChanReader` and `ChanWriter` adapters will be kept exactly as they are today.

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

Most of what's available in `std::os` today will move to `std::env`,
and the signatures will be updated to follow this RFC's
[Design principles] as follows.

**Arguments**:

* `args`: change to yield an iterator rather than vector if possible; in any case, it should produce an `OsStrBuf`.

**Environment variables**:

* `vars` (renamed from `env`): yields a vector of `(OsStrBuf, OsStrBuf)` pairs.
* `var` (renamed from `getenv`): take a value bounded by `IntoOsStrBuf`,
  allowing Rust strings and slices to be ergonomically passed in. Yields an `Option<OsStrBuf>`.
* `set_var` (renamed from `setenv`): takes two `IntoOsStrBuf`-bounded values.
* `remove_var` (renamed from `unsetenv`): takes a `IntoOsStrBuf`-bounded value.

* `join_paths`: take an `IntoIterator<T>` where `T: IntoOsStrBuf`, yield a `Result<OsString, JoinPathsError>`.
* `split_paths` take a `IntoOsStrBuf`, yield an `Iterator<Path>`.

**Working directory**:

* `current_dir` (renamed from `getcwd`): yields a `PathBuf`.
* `set_current_dir` (renamed from `change_dir`): takes an `AsPath` value.

**Important locations**:

* `home_dir` (renamed from `homedir`): returns home directory as a `PathBuf`
* `temp_dir` (renamed from `tmpdir`): returns a temporary directly as a `PathBuf`
* `current_exe` (renamed from `self_exe_name`): returns the full path
  to the current binary as a `PathBuf`.

**Exit status**:

* `get_exit_status` and `set_exit_status` stay as they are, but with
  updated docs that reflect that these only affect the return value of
  `std::rt::start`.

**Architecture information**:

* `num_cpus`, `page_size`: stay as they are

**Constants**:

* Stabilize `ARCH`, `DLL_PREFIX`, `DLL_EXTENSION`, `DLL_SUFFIX`, `EXE_EXTENSION`, `EXE_SUFFIX`, `FAMILY` as they are.
* Rename `SYSNAME` to `OS`.
* Remove `TMPBUF_SZ`.

This brings the constants into line with our naming conventions elsewhere.

#### Items to move to `os::platform`

* `pipe` will move to `os::unix`. It is currently primarily used for
  hooking to the IO of a child process, which will now be done behind
  a trait object abstraction.

#### Removed items

* `errno`, `error_string` and `last_os_error` provide redundant,
  platform-specific functionality and will be removed for now. They
  may reappear later in `os::unix` and `os::windows` in a modified
  form.
* `dll_filename`: deprecated in favor of working directly with the constants.
* `_NSGetArgc`, `_NSGetArgv`: these should never have been public.
* `self_exe_path`: deprecated in favor of `current_exe` plus path operations.
* `make_absolute`: deprecated in favor of explicitly joining with the working directory.
* all `_as_bytes` variants: deprecated in favor of yielding `OsStrBuf` values

### `std::fs`
[std::fs]: #stdfs

The `fs` module will provide most of the functionality it does today,
but with a stronger cross-platform orientation.

Note that all path-consuming functions will now take an
`AsPath`-bounded parameter for ergonomic reasons (this will allow
passing in Rust strings and literals directly, for example).

#### Free functions
[Free functions]: #free-functions

**Files**:

* `copy`. Take `AsPath` bound.
* `rename`. Take `AsPath` bound.
* `remove_file` (renamed from `unlink`). Take `AsPath` bound.

* `file_attr` (renamed from `stat`). Take `AsPath` bound. Yield a new
  struct, `FileAttr`, with no public fields, but `size`, `kind` and
  `perm` accessors. The various `os::platform` modules will offer
  extension methods on this structure.

* `set_perm` (renamed from `chmod`). Take `AsPath` bound, and a
  `FilePermissions` value. The `FilePermissions` type will be revamped
  as a struct with private implementation; see below.

**Directories**:

* `make_dir` (renamed from `mkdir`). Take `AsPath` bound.
* `make_dir_all` (renamed from `mkdir_recursive`). Take `AsPath` bound.
* `read_dir` (renamed from `readdir`). Take `AsPath` bound. Yield a
  newtypes iterator, which yields a new type `DirEntry` which has an
  accessor for `Path`, but will eventually provide other information
  as well (possibly via platform-specific extensions).
* `remove_dir` (renamed from `rmdir`). Take `AsPath` bound.
* `remove_dir_all` (renamed from `rmdir_recursive`). Take
  `AsPath` bound.
* `walk_dir`. Take `AsPath` bound. Yield an iterator over `IoResult<DirEntry>`.

**Links**:

* `hard_link` (renamed from `link`). Take `AsPath` bound.
* `sym_link` (renamed from `symlink`). Take `AsPath` bound.
* `read_link` (renamed form `readlink`). Take `AsPath` bound.

#### Files
[Files]: #files

The `File` type will largely stay as it is today, except that it will
use the `AsPath` bound everywhere.

The `stat` method will be renamed to `attr`, yield a `FileAttr`, and
take `&self`.

The `fsync` method will be renamed to `flush_os`, and `datasync` will
be moved to `os::unix` (since it has no meaning on Windows)

The `path` method wil remain `#[unstable]`, as we do not yet want to
commit to its API.

The `open_mode` function will take an `OpenOptions` struct, which will
encompass today's `FileMode` and `FileAccess` and support a
builder-style API.

#### File kinds
[File kinds]: #file-kinds

The `FileType` module will be renamed to `FileKind`, and the
underlying `enum` will be hidden (to allow for platform differences
and growth). It will expose at least `is_file` and `is_dir`; the other
methods need to be audited for compatibility across
platforms. Platform-specific kinds will be relegated to extension
traits in `std::os::platform`.

#### File permissions
[File permissions]: #file-permissions

Unfortunately, the permission models on Unix and Windows vary
greatly. Rather than offer an API that has no meaning on some
platforms, we will provide a very limited `FilePermissions` structure
in `std::fs`, and then rich extension traits in `std::os::unix` and
`std::os::windows`.

On the Unix side, the constructors and accessors for `FilePermissions`
will resemble the flags we have today; details are left to the implementation.

On the Windows side, initially there will be no extensions, as Windows
has a very complex permissions model that will take some time to build
out.

For `std::fs` itself, `FilePermissions` will provide constructors and
accessors for "world readable" -- and that is all. At the moment, that
is all that is known to be compatible across the platforms that Rust
supports.

#### `PathExt`
[PathExt]: #pathext

This trait will essentially remain stay as it is (renamed from
`PathExtensions`), following the same changes made to `fs` free functions.

#### Items to move to `os::platform`

* `change_file_times` will move to `os::unix` for now (cf
  `SetFileTime` on Windows). Eventually we will add back a
  cross-platform function, when we have grown a notion of time in
  `std` and have a good compatibility story across all platforms.

* `lstat` will move to `os::unix` since it is not yet implemented for
  Windows.

* `chown` will move to `os::unix` (it currently does *nothing* on
  Windows), and eventually `os::windows` will grow support for
  Windows's permission model. If at some point a reasonable
  intersection is found, we will re-introduce a cross-platform
  function in `std::fs`.

* In general, offer all of the `stat` fields as an extension trait on
  `FileAttr` (e.g. `os::unix::FileAttrExt`).

### `std::net`
[std::net]: #stdnet

The contents of `std::io::net` submodules `tcp`, `udp`, `ip` and
`addrinfo` will be retained but moved into a single `std::net` module;
the other modules are being moved or removed and are described
elsewhere.

#### TCP
[TCP]: #tcp

For `TcpStream`, the changes are most easily expressed by giving the signatures directly:

```rust
// TcpStream, which contains both a reader and a writer

impl TcpStream {
    fn connect<A: ToSocketAddr>(addr: A) -> IoResult<TcpStreama>;
    fn connect_deadline<A: ToSocketAddr>(addr: A, deadline: Duration) -> IoResult<TcpStreama>;

    fn reader(&mut self) -> &mut TcpReader;
    fn writer(&mut self) -> &mut TcpWriter;
    fn split(self) -> (TcpReader, TcpWriter);

    fn peer_addr(&mut self) -> IoResult<SocketAddr>;
    fn socket_addr(&mut self) -> IoResult<SocketAddr>;
}

impl Reader for TcpStream { ... }
impl Writer for TcpStream { ... }

impl Reader for Deadlined<TcpStream> { ... }
impl Writer for Deadlined<TcpStream> { ... }

// TcpReader

impl Reader for TcpReader { ... }
impl Reader for Deadlined<TcpReader> { ... }

impl TcpReader {
    fn peer_addr(&mut self) -> IoResult<SocketAddr>;
    fn socket_addr(&mut self) -> IoResult<SocketAddr>;

    fn shutdown_token(&mut self) -> ShutdownToken;
}

// TcpWriter

impl Writer for TcpWriter { ... }
impl Writer for Deadlined<TcpWriter> { ... }

impl TcpWriter {
    fn peer_addr(&mut self) -> IoResult<SocketAddr>;
    fn socket_addr(&mut self) -> IoResult<SocketAddr>;

    fn shutdown_token(&mut self) -> ShutdownToken;
}

// ShutdownToken

impl ShutdownToken {
    fn shutdown(self);
}

impl Clone for ShutdownToken { ... }
```

The idea is that a `TcpStream` provides both a reader and a writer,
and can be used directly as such, just as it can today. However, the
two sides can also be broken apart via the `split` method, which
allows them to be shipped off to separate threads. Moreover, each side
can yield a `ShutdownToken`, a `Clone` and `Send` value that can be
used to shut down that side of the socket, cancelling any in-progress
blocking operations, much like e.g. `close_read` does today.

The implementation of the `ShutdownToken` infrastructure should ensure
that there is essentially no cost imposed when the feature is not used
-- in particular, if a `ShutdownToken` has not been requested, a
single `read` or `write` should correspond to a single syscall.

For `TcpListener`, the only change is to rename `socket_name` to
`socket_addr`.

For `TcpAcceptor` we will:

* Add a `socket_addr` method.
* Possibly provide a convenience constructor for `bind`.
* Replace `close_accept` with `cancel_token()`.
* Remove `Clone`.
* Rename `IncomingConnecitons` to `Incoming`.

#### UDP
[UDP]: #udp

The UDP infrastructure should change to use the new deadline
infrastructure, but should not provide `Clone`, `ShutdownToken`s, or a
reader/writer split. In addition:

* `recv_from` should become `recv`.
* `send_to` should become `send`.
* `socket_name` should become `socket_addr`.

Methods like `multicast` and `ttl` are left as `#[experimental]` for
now (they are derived from libuv's design).

#### Addresses
[Addresses]: #addresses

For the current `addrinfo` module:

* The `get_host_addresses` should be renamed to `lookup_host`.
* All other contents should be removed.

For the current `ip` module:

* The `ToSocketAddr` trait should become `ToSocketAddrs`
* The default `to_socket_addr_all` method should be removed.

The actual address structures could use some scrutiny, but any
revisions there are left as an unresolved question.

### `std::process`
[std::process]: #stdprocess

Currently `std::io::process` is used only for spawning new
processes. The re-envisioned `std::process` will ultimately support
inspecting currently-running processes, although this RFC does not
propose any immediate support for doing so -- it merely future-proofs
the module.

#### `Command`
[Command]: #command

The `Command` type is a builder API for processes, and is largely in
good shape, modulo a few tweaks:

* Replace `ToCCstr` bounds with `IntoOsStrBuf`.
* Replace `env_set_all` with `env_clear`
* Rename `cwd` to `current_dir`, take `AsPath`.
* Rename `spawn` to `run`
* Move `uid` and `gid` to an extension trait in `os::unix`
* Make `detached` take a `bool` (rather than always setting the
  command to detached mode).

The `stdin`, `stdout`, `stderr` methods will undergo a more
significant change. By default, the corresponding options we be
considered "unset", the interpretation of which depends on how the
process is launched:

* For `run` or `status`, these will inherit from the current process by default.
* For `output`, these will capture to new readers/writers by default.

The `StdioContainer` type will be renamed to `Stdio`, and will not be
exposed directly as an enum (to enable growth and change over time).
It will provide a `Capture` constructor for capturing input or output,
an `Inherit` constructor (which just means to use the current IO
object -- it does not take an argument), and a `Null` constructor. The
equivalent of today's `InheritFd` will be added at a later point.

#### `Child`
[Child]: #child

We propose renaming `Process` to `Child` so that we can add a
more general notion of non-child `Process` later on (every
`Child` will be able to give you a `Process`).

* `stdin`, `stdout` and `stderr` will be retained as public fields,
  but their types will change to `Box<Reader+Send>` or
  `Box<Writer+Send>` as appropriate. This effectively hides the internal
  pipe infrastructure.
* The `kill` method is dropped, and `id` and `signal` will move to `os::platform` extension traits.
* `signal_exit`, `signal_kill`, `wait`, and `forget` will all stay as they are.
* `wait_with_output` will take `&self`.
* `set_timeout` will be changed to use the `with_deadline` infrastructure.

There are also a few other related changes to the module:

* Rename `ProcessOuptput` to `Output`
* Rename `ProcessExit` to `ExitStatus`, and hide its
  representation. Remove `matches_exit_status`, and add a `status`
  method yielding an `Option<i32>
* Remove `MustDieSignal`, `PleaseExitSignal`.
* Remove `EnvMap` (which should never have been exposed).

### `std::os`
[std::os]: #stdos

Initially, this module will be empty except for the platform-specific
`unix` and `windows` modules. It is expected to grow additional, more
specific platform submodules (like `linux`, `macos`) over time.

## Odds and ends
[Odds and ends]: #odds-and-ends

### The `io` prelude
[The io prelude]: #the-io-prelude

The `prelude` submodule will contain most of the traits, types, and
modules discussed in this RFC; it is meant to provide maximal
convenience when working with IO of any kind. The exact contents of
the module are left as an open question.

### Functionality moved elsewhere
[Functionality moved elsewhere]: #functionality-moved-elsewhere

* The `set_stdout` and `set_stderr` will be moved to a new
  `std::fmt::output` submodule and renamed `set_print` and
  `set_panic`, respectively. These new names reflect what the
  functions actually do, removing a longstanding confusion. A
  `flush_print` method will also be added to the same module.

* The `std::io::net::pipe` module will move to `os::platform` modules,
  removing the rather artificial "cross-platform" support currently
  provides.

* The `std::os::MemoryMap` type will move to `os::platform` modules
  that can evolve independently.

### Functionality removed outright
[Functionality removed outright]: #functionality-removed-outright

* `io::Acceptor`, `io::Listener`. These traits are not terribly useful
  as an abstraction right now, and can always be incorporated in a
  more useful form later on. (This is especially true with
  `UnixStream` moving into `os::unix`.)
* `io::Stream`. This alias serves little purpose at the moment.
* `io::timer`. This module will be removed outright, and the `sleep`
  function will move to `std::thread::Thread`.
* `io::test`. Removed.
* `io::pipe`. Removed in favor of returning `Box<Reader+Send>` or
  `Box<Writer+Send>` for talking to spawned processes.

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

* What precisely should `std::io::prelude` contain?
* The detailed design of the `OpenOptions` builder.
* The fate of `stdin_raw` and `stdout_raw`.
* Are `IpAddr` and `SocketAddr` complete? If not, should their
  representation be hidden so that it can be extended later?
