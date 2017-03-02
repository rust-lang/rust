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
    * [String handling]
        * [Key observations]
        * [The design: `os_str`]
        * [The future]
    * [Deadlines] (stub)
    * [Splitting streams and cancellation] (stub)
    * [Modules]
        * [core::io]
            * [Adapters]
            * [Free functions]
            * [Seeking]
            * [Buffering]
            * [Cursor]
        * [The std::io facade]
            * [Errors]
            * [Channel adapters]
            * [stdin, stdout, stderr]
            * [Printing functions]
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

Existing blocking APIs all have to deal with this problem, and Rust
can and should follow the existing tradition here. See
[Revising `Reader` and `Writer`] for the proposed solution.

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
is also not highly ergonomic as presently implemented. (Consider `os::getenv`
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
  cut down on the number of non-atomic read operations and (4) adjust
  the remaining operations to provide more flexibility when possible.

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

With those general points out of the way, let's look at the details.

### `Read`
[Read]: #read

The updated `Reader` trait (and its extension) is as follows:

```rust
trait Read {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Error>;

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<(), Error> { ... }
    fn read_to_string(&self, buf: &mut String) -> Result<(), Error> { ... }
}

// extension trait needed for object safety
trait ReadExt: Read {
    fn bytes(&mut self) -> Bytes<Self> { ... }

    ... // more to come later in the RFC
}
impl<R: Read> ReadExt for R {}
```

Following the
[trait naming conventions](https://github.com/rust-lang/rfcs/pull/344),
the trait is renamed to `Read` reflecting the clear primary method it
provides.

The `read` method should not involve internal looping (even over
errors like `EINTR`). It is intended to faithfully represent a single
call to an underlying system API.

The `read_to_end` and `read_to_string` methods now take explicit
buffers as input. This has multiple benefits:

* Performance. When it is known that reading will involve some large
  number of bytes, the buffer can be preallocated in advance.

* "Atomicity" concerns. For `read_to_end`, it's possible to use this
  API to retain data collected so far even when a `read` fails in the
  middle. For `read_to_string`, this is not the case, because UTF-8
  validity cannot be ensured in such cases; but if intermediate
  results are wanted, one can use `read_to_end` and convert to a
  `String` only at the end.

Convenience methods like these will retry on `EINTR`. This is partly
under the assumption that in practice, EINTR will *most often* arise
when interfacing with other code that changes a signal handler. Due to
the global nature of these interactions, such a change can suddenly
cause your own code to get an error irrelevant to it, and the code
should probably just retry in those cases. In the case where you are
using EINTR explicitly, `read` and `write` will be available to handle
it (and you can always build your own abstractions on top).

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
    fn write(&mut self, buf: &[u8]) -> Result<uint, Error>;
    fn flush(&mut self) -> Result<(), Error>;

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Error> { .. }
    fn write_fmt(&mut self, fmt: &fmt::Arguments) -> Result<(), Error> { .. }
}
```

The biggest change here is to the semantics of `write`. Instead of
repeatedly writing to the underlying IO object until all of `buf` is
written, it attempts a *single* write and on success returns the
number of bytes written. This follows the long tradition of blocking
IO, and is a more fundamental building block than the looping write we
currently have. Like `read`, it will propagate EINTR.

For convenience, `write_all` recovers the behavior of today's `write`,
looping until either the entire buffer is written or an error
occurs. To meaningfully recover from an intermediate error and keep
writing, code should work with `write` directly. Like the `Read`
conveniences, `EINTR` results in a retry.

The `write_fmt` method, like `write_all`, will loop until its entire
input is written or an error occurs.

The other methods include endian conversions (covered by
serialization) and a few conveniences like `write_str` for other basic
types. The latter, at least, is already uniformly (and extensibly)
covered via the `write!` macro. The other helpers, as with `Read`,
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
  often called UCS-2; see http://justsolve.archiveteam.org/wiki/UCS-2 for a bit
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
  my_ut8_data.to_wtf8().to_ucs2().as_u16_slice() == my_utf8_data.to_utf16().as_u16_slice()
  ```

* Valid UTF-16 data re-encoded as WTF-8 produces the corresponding UTF-8 data:

  ```rust
  my_utf16_data.to_wtf8().as_bytes() == my_utf16_data.to_utf8().as_bytes()
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

The `OsString` type is analogous to `String`, and `OsStr` is analogous to `str`.
Their backing implementation is platform-dependent, but they offer a
cross-platform API:

```rust
pub mod os_str {
    /// Owned OS strings
    struct OsString {
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
        type Slice = [u8];
        ...
    }

    #[cfg(windows)]
    mod imp {
        type Buf = Wtf8Buf; // See https://github.com/SimonSapin/rust-wtf8
        type Slice = Wtf8;
        ...
    }

    impl OsString {
        pub fn from_string(String) -> OsString;
        pub fn from_str(&str) -> OsString;
        pub fn as_slice(&self) -> &OsStr;
        pub fn into_string(Self) -> Result<String, OsString>;
        pub fn into_string_lossy(Self) -> String;

        // and ultimately other functionality typically found on vectors,
        // but CRUCIALLY NOT as_bytes
    }

    impl Deref<OsStr> for OsString { ... }

    impl OsStr {
        pub fn from_str(value: &str) -> &OsStr;
        pub fn as_str(&self) -> Option<&str>;
        pub fn to_string_lossy(&self) -> CowString;

        // and ultimately other functionality typically found on slices,
        // but CRUCIALLY NOT as_bytes
    }

    trait IntoOsString {
        fn into_os_str_buf(self) -> OsString;
    }

    impl IntoOsString for OsString { ... }
    impl<'a> IntoOsString for &'a OsStr { ... }

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
        trait OsStringExt {
            fn from_vec(Vec<u8>) -> Self;
            fn into_vec(Self) -> Vec<u8>;
        }

        impl OsStringExt for os_str::OsString { ... }

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

        trait OsStringExt {
            fn from_wide_slice(&[u16]) -> Self;
        }

        impl OsStringExt for os_str::OsString { ... }

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

Ideally, the `io` module will be split into the parts that can live in
`libcore` (most of it) and the parts that are added in the `std::io`
facade. This part of the organization is non-normative, since it
requires changes to today's `IoError` (which currently references
`String`); if these changes cannot be performed, everything here will
live in `std::io`.

#### Adapters
[Adapters]: #adapters

The current `std::io::util` module offers a number of `Reader` and
`Writer` "adapters". This RFC refactors the design to more closely
follow `std::iter`. Along the way, it generalizes the `by_ref` adapter:

```rust
trait ReadExt: Read {
    // ... eliding the methods already described above

    // Postfix version of `(&mut self)`
    fn by_ref(&mut self) -> &mut Self { ... }

    // Read everything from `self`, then read from `next`
    fn chain<R: Read>(self, next: R) -> Chain<Self, R> { ... }

    // Adapt `self` to yield only the first `limit` bytes
    fn take(self, limit: u64) -> Take<Self> { ... }

    // Whenever reading from `self`, push the bytes read to `out`
    #[unstable] // uncertain semantics of errors "halfway through the operation"
    fn tee<W: Write>(self, out: W) -> Tee<Self, W> { ... }
}

trait WriteExt: Write {
    // Postfix version of `(&mut self)`
    fn by_ref<'a>(&'a mut self) -> &mut Self { ... }

    // Whenever bytes are written to `self`, write them to `other` as well
    #[unstable] // uncertain semantics of errors "halfway through the operation"
    fn broadcast<W: Write>(self, other: W) -> Broadcast<Self, W> { ... }
}

// An adaptor converting an `Iterator<u8>` to `Read`.
pub struct IterReader<T> { ... }
```

As with `std::iter`, these adapters are object unsafe and hence placed
in an extension trait with a blanket `impl`.

#### Free functions
[Free functions]: #free-functions

The current `std::io::util` module also includes a number of primitive
readers and writers, as well as `copy`. These are updated as follows:

```rust
// A reader that yields no bytes
fn empty() -> Empty; // in theory just returns `impl Read`

impl Read for Empty { ... }

// A reader that yields `byte` repeatedly (generalizes today's ZeroReader)
fn repeat(byte: u8) -> Repeat;

impl Read for Repeat { ... }

// A writer that ignores the bytes written to it (/dev/null)
fn sink() -> Sink;

impl Write for Sink { ... }

// Copies all data from a `Read` to a `Write`, returning the amount of data
// copied.
pub fn copy<R, W>(r: &mut R, w: &mut W) -> Result<u64, Error>
```

Like `write_all`, the `copy` method will discard the amount of data already
written on any error and also discard any partially read data on a `write`
error. This method is intended to be a convenience and `write` should be used
directly if this is not desirable.

#### Seeking
[Seeking]: #seeking

The seeking infrastructure is largely the same as today's, except that
`tell` is removed and the `seek` signature is refactored with more precise
types:

```rust
pub trait Seek {
    // returns the new position after seeking
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Error>;
}

pub enum SeekFrom {
    Start(u64),
    End(i64),
    Current(i64),
}
```

The old `tell` function can be regained via `seek(SeekFrom::Current(0))`.

#### Buffering
[Buffering]: #buffering

The current `Buffer` trait will be renamed to `BufRead` for
clarity (and to open the door to `BufWrite` at some later
point):

```rust
pub trait BufRead: Read {
    fn fill_buf(&mut self) -> Result<&[u8], Error>;
    fn consume(&mut self, amt: uint);

    fn read_until(&mut self, byte: u8, buf: &mut Vec<u8>) -> Result<(), Error> { ... }
    fn read_line(&mut self, buf: &mut String) -> Result<(), Error> { ... }
}

pub trait BufReadExt: BufRead {
    // Split is an iterator over Result<Vec<u8>, Error>
    fn split(&mut self, byte: u8) -> Split<Self> { ... }

    // Lines is an iterator over Result<String, Error>
    fn lines(&mut self) -> Lines<Self> { ... };

    // Chars is an iterator over Result<char, Error>
    fn chars(&mut self) -> Chars<Self> { ... }
}
```

The `read_until` and `read_line` methods are changed to take explicit,
mutable buffers, for similar reasons to `read_to_end`. (Note that
buffer reuse is particularly common for `read_line`). These functions
include the delimiters in the strings they produce, both for easy
cross-platform compatibility (in the case of `read_line`) and for ease
in copying data without loss (in particular, distinguishing whether
the last line included a final delimiter).

The `split` and `lines` methods provide iterator-based versions of
`read_until` and `read_line`, and *do not* include the delimiter in
their output. This matches conventions elsewhere (like `split` on
strings) and is usually what you want when working with iterators.

The `BufReader`, `BufWriter` and `BufStream` types stay
essentially as they are today, except that for streams and writers the
`into_inner` method yields the structure back in the case of a write error,
and its behavior is clarified to writing out the buffered data without
flushing the underlying reader:
```rust
// If writing fails, you get the unwritten data back
fn into_inner(self) -> Result<W, IntoInnerError<Self>>;

pub struct IntoInnerError<W>(W, Error);

impl IntoInnerError<T> {
    pub fn error(&self) -> &Error { ... }
    pub fn into_inner(self) -> W { ... }
}
impl<W> FromError<IntoInnerError<W>> for Error { ... }
```

#### `Cursor`
[Cursor]: #cursor

Many applications want to view in-memory data as either an implementor of `Read`
or `Write`. This is often useful when composing streams or creating test cases.
This functionality primarily comes from the following implementations:

```rust
impl<'a> Read for &'a [u8] { ... }
impl<'a> Write for &'a mut [u8] { ... }
impl Write for Vec<u8> { ... }
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

impl Seek for Cursor<Vec<u8>> { ... }
impl<'a> Seek for Cursor<&'a [u8]> { ... }
impl<'a> Seek for Cursor<&'a mut [u8]> { ... }

impl Read for Cursor<Vec<u8>> { ... }
impl<'a> Read for Cursor<&'a [u8]> { ... }
impl<'a> Read for Cursor<&'a mut [u8]> { ... }

impl BufRead for Cursor<Vec<u8>> { ... }
impl<'a> BufRead for Cursor<&'a [u8]> { ... }
impl<'a> BufRead for Cursor<&'a mut [u8]> { ... }

impl<'a> Write for Cursor<&'a mut [u8]> { ... }
impl Write for Cursor<Vec<u8>> { ... }
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

The current `stdio` module will be removed in favor of these constructors in the
`io` module:

```rust
pub fn stdin() -> Stdin;
pub fn stdout() -> Stdout;
pub fn stderr() -> Stderr;
```

* `stdin` - returns a handle to a **globally shared** standard input of
  the process which is buffered as well. Due to the globally shared nature of
  this handle, all operations on `Stdin` directly will acquire a lock internally
  to ensure access to the shared buffer is synchronized. This implementation
  detail is also exposed through a `lock` method where the handle can be
  explicitly locked for a period of time so relocking is not necessary.

  The `Read` trait will be implemented directly on the returned `Stdin` handle
  but the `BufRead` trait will not be (due to synchronization concerns). The
  locked version of `Stdin` (`StdinLock`) will provide an implementation of
  `BufRead`.

  The design will largely be the same as is today with the `old_io` module.

  ```rust
  impl Stdin {
      fn lock(&self) -> StdinLock;
      fn read_line(&mut self, into: &mut String) -> io::Result<()>;
      fn read_until(&mut self, byte: u8, into: &mut Vec<u8>) -> io::Result<()>;
  }
  impl Read for Stdin { ... }
  impl Read for StdinLock { ... }
  impl BufRead for StdinLock { ... }
  ```

* `stderr` - returns a **non buffered** handle to the standard error output
  stream for the process. Each call to `write` will roughly translate to a
  system call to output data when written to `stderr`. This handle is locked
  like `stdin` to ensure, for example, that calls to `write_all` are atomic with
  respect to one another. There will also be an RAII guard to lock the handle
  and use the result as an instance of `Write`.

  ```rust
  impl Stderr {
      fn lock(&self) -> StderrLock;
  }
  impl Write for Stderr { ... }
  impl Write for StderrLock { ... }
  ```

* `stdout` - returns a **globally buffered** handle to the standard output of
  the current process. The amount of buffering can be decided at runtime to
  allow for different situations such as being attached to a TTY or being
  redirected to an output file. The `Write` trait will be implemented for this
  handle, and like `stderr` it will be possible to lock it and then use the
  result as an instance of `Write` as well.

  ```rust
  impl Stdout {
      fn lock(&self) -> StdoutLock;
  }
  impl Write for Stdout { ... }
  impl Write for StdoutLock { ... }
  ```

#### Windows and stdio
[Windows stdio]: #windows-and-stdio

On Windows, standard input and output handles can work with either arbitrary
`[u8]` or `[u16]` depending on the state at runtime. For example a program
attached to the console will work with arbitrary `[u16]`, but a program attached
to a pipe would work with arbitrary `[u8]`.

To handle this difference, the following behavior will be enforced for the
standard primitives listed above:

* If attached to a pipe then no attempts at encoding or decoding will be done,
  the data will be ferried through as `[u8]`.

* If attached to a console, then `stdin` will attempt to interpret all input as
  UTF-16, re-encoding into UTF-8 and returning the UTF-8 data instead. This
  implies that data will be buffered internally to handle partial reads/writes.
  Invalid UTF-16 will simply be discarded returning an `io::Error` explaining
  why.

* If attached to a console, then `stdout` and `stderr` will attempt to interpret
  input as UTF-8, re-encoding to UTF-16. If the input is not valid UTF-8 then an
  error will be returned and no data will be written.

#### Raw stdio
[Raw stdio]: #raw-stdio

> **Note**: This section is intended to be a sketch of possible raw stdio
>           support, but it is not planned to implement or stabilize this
>           implementation at this time.

The above standard input/output handles all involve some form of locking or
buffering (or both). This cost is not always wanted, and hence raw variants will
be provided. Due to platform differences across unix/windows, the following
structure will be supported:

```rust
mod os {
    mod unix {
        mod stdio {
            struct Stdio { .. }

            impl Stdio {
                fn stdout() -> Stdio;
                fn stderr() -> Stdio;
                fn stdin() -> Stdio;
            }

            impl Read for Stdio { ... }
            impl Write for Stdio { ... }
        }
    }

    mod windows {
        mod stdio {
            struct Stdio { ... }
            struct StdioConsole { ... }

            impl Stdio {
                fn stdout() -> io::Result<Stdio>;
                fn stderr() -> io::Result<Stdio>;
                fn stdin() -> io::Result<Stdio>;
            }
            // same constructors StdioConsole

            impl Read for Stdio { ... }
            impl Write for Stdio { ... }

            impl StdioConsole {
                // returns slice of what was read
                fn read<'a>(&self, buf: &'a mut OsString) -> io::Result<&'a OsStr>;
                // returns remaining part of `buf` to be written
                fn write<'a>(&self, buf: &'a OsStr) -> io::Result<&'a OsStr>;
            }
        }
    }
}
```

There are some key differences from today's API:

* On unix, the API has not changed much except that the handles have been
  consolidated into one type which implements both `Read` and `Write` (although
  writing to stdin is likely to generate an error).
* On windows, there are two sets of handles representing the difference between
  "console mode" and not (e.g. a pipe). When not a console the normal I/O traits
  are implemented (delegating to `ReadFile` and `WriteFile`. The console mode
  operations work with `OsStr`, however, to show how they work with UCS-2 under
  the hood.

#### Printing functions
[Printing functions]: #printing-functions

The current `print`, `println`, `print_args`, and `println_args` functions will
all be "removed from the public interface" by [prefixing them with `__` and
marking `#[doc(hidden)]`][gh22607]. These are all implementation details of the
`print!` and `println!` macros and don't need to be exposed in the public
interface.

[gh22607]: https://github.com/rust-lang/rust/issues/22607

The `set_stdout` and `set_stderr` functions will be removed with no replacement
for now. It's unclear whether these functions should indeed control a thread
local handle instead of a global handle as whether they're justified in the
first place. It is a backwards-compatible extension to allow this sort of output
to be redirected and can be considered if the need arises.

### `std::env`
[std::env]: #stdenv

Most of what's available in `std::os` today will move to `std::env`,
and the signatures will be updated to follow this RFC's
[Design principles] as follows.

**Arguments**:

* `args`: change to yield an iterator rather than vector if possible; in any
  case, it should produce an `OsString`.

**Environment variables**:

* `vars` (renamed from `env`): yields a vector of `(OsString, OsString)` pairs.
* `var` (renamed from `getenv`): take a value bounded by `AsOsStr`,
  allowing Rust strings and slices to be ergonomically passed in. Yields an
  `Option<OsString>`.
* `var_string`: take a value bounded by `AsOsStr`, returning `Result<String,
  VarError>` where `VarError` represents a non-unicode `OsString` or a "not
  present" value.
* `set_var` (renamed from `setenv`): takes two `AsOsStr`-bounded values.
* `remove_var` (renamed from `unsetenv`): takes a `AsOsStr`-bounded value.

* `join_paths`: take an `IntoIterator<T>` where `T: AsOsStr`, yield a
  `Result<OsString, JoinPathsError>`.
* `split_paths` take a `AsOsStr`, yield an `Iterator<Path>`.

**Working directory**:

* `current_dir` (renamed from `getcwd`): yields a `PathBuf`.
* `set_current_dir` (renamed from `change_dir`): takes an `AsPath` value.

**Important locations**:

* `home_dir` (renamed from `homedir`): returns home directory as a `PathBuf`
* `temp_dir` (renamed from `tmpdir`): returns a temporary directly as a `PathBuf`
* `current_exe` (renamed from `self_exe_name`): returns the full path
  to the current binary as a `PathBuf` in an `io::Result` instead of an
  `Option`.

**Exit status**:

* `get_exit_status` and `set_exit_status` stay as they are, but with
  updated docs that reflect that these only affect the return value of
  `std::rt::start`. These will remain `#[unstable]` for now and a future RFC
  will determine their stability.

**Architecture information**:

* `num_cpus`, `page_size`: stay as they are, but remain `#[unstable]`. A future
  RFC will determine their stability and semantics.

**Constants**:

* Stabilize `ARCH`, `DLL_PREFIX`, `DLL_EXTENSION`, `DLL_SUFFIX`,
  `EXE_EXTENSION`, `EXE_SUFFIX`, `FAMILY` as they are.
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
* all `_as_bytes` variants: deprecated in favor of yielding `OsString` values

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

* `metadata` (renamed from `stat`). Take `AsPath` bound. Yield a new
  struct, `Metadata`, with no public fields, but `len`, `is_dir`,
  `is_file`, `perms`, `accessed` and `modified` accessors. The various
  `os::platform` modules will offer extension methods on this
  structure.

* `set_perms` (renamed from `chmod`). Take `AsPath` bound, and a
  `Perms` value. The `Perms` type will be revamped
  as a struct with private implementation; see below.

**Directories**:

* `create_dir` (renamed from `mkdir`). Take `AsPath` bound.
* `create_dir_all` (renamed from `mkdir_recursive`). Take `AsPath` bound.
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
* `soft_link` (renamed from `symlink`). Take `AsPath` bound.
* `read_link` (renamed form `readlink`). Take `AsPath` bound.

#### Files
[Files]: #files

The `File` type will largely stay as it is today, except that it will
use the `AsPath` bound everywhere.

The `stat` method will be renamed to `metadata`, yield a `Metadata`
structure (as described above), and take `&self`.

The `fsync` method will be renamed to `sync_all`, and `datasync` will be
renamed to `sync_data`. (Although the latter is not available on
Windows, it can be considered an optimization for `flush` and on
Windows behave identically to `sync_all`, just as it does on some Unix
filesystems.)

The `path` method wil remain `#[unstable]`, as we do not yet want to
commit to its API.

The `open_mode` function will be removed in favor of and will take an
`OpenOptions` struct, which will encompass today's `FileMode` and
`FileAccess` and support a builder-style API.

#### File kinds
[File kinds]: #file-kinds

The `FileType` type will be removed. As mentioned above, `is_file` and
`is_dir` will be provided directly on `Metadata`; the other types
need to be audited for compatibility across
platforms. Platform-specific kinds will be relegated to extension
traits in `std::os::platform`.

It's possible that an
[extensible](https://github.com/rust-lang/rfcs/pull/757) `Kind` will
be added in the future.

#### File permissions
[File permissions]: #file-permissions

The permission models on Unix and Windows vary greatly -- even between
different filesystems within the same OS. Rather than offer an API
that has no meaning on some platforms, we will initially provide a
very limited `Perms` structure in `std::fs`, and then rich
extension traits in `std::os::unix` and `std::os::windows`. Over time,
if clear cross-platform patterns emerge for richer permissions, we can
grow the `Perms` structure.

On the Unix side, the constructors and accessors for `Perms`
will resemble the flags we have today; details are left to the implementation.

On the Windows side, initially there will be no extensions, as Windows
has a very complex permissions model that will take some time to build
out.

For `std::fs` itself, `Perms` will provide constructors and
accessors for "world readable" -- and that is all. At the moment, that
is all that is known to be compatible across the platforms that Rust
supports.

#### `PathExt`
[PathExt]: #pathext

This trait will essentially remain stay as it is (renamed from
`PathExtensions`), following the same changes made to `fs` free functions.

#### Items to move to `os::platform`

* `lstat` will move to `os::unix` and remain `#[unstable]` *for now*
  since it is not yet implemented for Windows.

* `chown` will move to `os::unix` (it currently does *nothing* on
  Windows), and eventually `os::windows` will grow support for
  Windows's permission model. If at some point a reasonable
  intersection is found, we will re-introduce a cross-platform
  function in `std::fs`.

* In general, offer all of the `stat` fields as an extension trait on
  `Metadata` (e.g. `os::unix::MetadataExt`).

### `std::net`
[std::net]: #stdnet

The contents of `std::io::net` submodules `tcp`, `udp`, `ip` and
`addrinfo` will be retained but moved into a single `std::net` module;
the other modules are being moved or removed and are described
elsewhere.

#### SocketAddr

This structure will represent either a `sockaddr_in` or `sockaddr_in6` which is
commonly just a pairing of an IP address and a port.

```rust
enum SocketAddr {
    V4(SocketAddrV4),
    V6(SocketAddrV6),
}

impl SocketAddrV4 {
    fn new(addr: Ipv4Addr, port: u16) -> SocketAddrV4;
    fn ip(&self) -> &Ipv4Addr;
    fn port(&self) -> u16;
}

impl SocketAddrV6 {
    fn new(addr: Ipv6Addr, port: u16, flowinfo: u32, scope_id: u32) -> SocketAddrV6;
    fn ip(&self) -> &Ipv6Addr;
    fn port(&self) -> u16;
    fn flowinfo(&self) -> u32;
    fn scope_id(&self) -> u32;
}
```

#### Ipv4Addr

Represents a version 4 IP address. It has the following interface:

```rust
impl Ipv4Addr {
    fn new(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr;
    fn any() -> Ipv4Addr;
    fn octets(&self) -> [u8; 4];
    fn to_ipv6_compatible(&self) -> Ipv6Addr;
    fn to_ipv6_mapped(&self) -> Ipv6Addr;
}
```

#### Ipv6Addr

Represents a version 6 IP address. It has the following interface:

```rust
impl Ipv6Addr {
    fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Ipv6Addr;
    fn any() -> Ipv6Addr;
    fn segments(&self) -> [u16; 8]
    fn to_ipv4(&self) -> Option<Ipv4Addr>;
}
```

#### TCP
[TCP]: #tcp

The current `TcpStream` struct will be pared back from where it is today to the
following interface:

```rust
// TcpStream, which contains both a reader and a writer

impl TcpStream {
    fn connect<A: ToSocketAddrs>(addr: &A) -> io::Result<TcpStream>;
    fn peer_addr(&self) -> io::Result<SocketAddr>;
    fn local_addr(&self) -> io::Result<SocketAddr>;
    fn shutdown(&self, how: Shutdown) -> io::Result<()>;
    fn try_clone(&self) -> io::Result<TcpStream>;
}

impl Read for TcpStream { ... }
impl Write for TcpStream { ... }
impl<'a> Read for &'a TcpStream { ... }
impl<'a> Write for &'a TcpStream { ... }
#[cfg(unix)]    impl AsRawFd for TcpStream { ... }
#[cfg(windows)] impl AsRawSocket for TcpStream { ... }
```

* `clone` has been replaced with a `try_clone` function. The implementation of
  `try_clone` will map to using `dup` on Unix platforms and
  `WSADuplicateSocket` on Windows platforms. The `TcpStream` itself will no
   longer be reference counted itself under the hood.
* `close_{read,write}` are both removed in favor of binding the `shutdown`
  function directly on sockets. This will map to the `shutdown` function on both
  Unix and Windows.
* `set_timeout` has been removed for now (as well as other timeout-related
  functions). It is likely that this may come back soon as a binding to
  `setsockopt` to the `SO_RCVTIMEO` and `SO_SNDTIMEO` options. This RFC does not
  currently proposed adding them just yet, however.
* Implementations of `Read` and `Write` are provided for `&TcpStream`. These
  implementations are not necessarily ergonomic to call (requires taking an
  explicit reference), but they express the ability to concurrently read and
  write from a `TcpStream`

Various other options such as `nodelay` and `keepalive` will be left
`#[unstable]` for now. The `TcpStream` structure will also adhere to both `Send`
and `Sync`.

The `TcpAcceptor` struct will be removed and all functionality will be folded
into the `TcpListener` structure. Specifically, this will be the resulting API:

```rust
impl TcpListener {
    fn bind<A: ToSocketAddrs>(addr: &A) -> io::Result<TcpListener>;
    fn local_addr(&self) -> io::Result<SocketAddr>;
    fn try_clone(&self) -> io::Result<TcpListener>;
    fn accept(&self) -> io::Result<(TcpStream, SocketAddr)>;
    fn incoming(&self) -> Incoming;
}

impl<'a> Iterator for Incoming<'a> {
    type Item = io::Result<TcpStream>;
    ...
}
#[cfg(unix)]    impl AsRawFd for TcpListener { ... }
#[cfg(windows)] impl AsRawSocket for TcpListener { ... }
```

Some major changes from today's API include:

* The static distinction between `TcpAcceptor` and `TcpListener` has been
  removed (more on this in the [socket][Sockets] section).
* The `clone` functionality has been removed in favor of `try_clone` (same
  caveats as `TcpStream`).
* The `close_accept` functionality is removed entirely. This is not currently
  implemented via `shutdown` (not supported well across platforms) and is
  instead implemented via `select`. This functionality can return at a later
  date with a more robust interface.
* The `set_timeout` functionality has also been removed in favor of returning at
  a later date in a more robust fashion with `select`.
* The `accept` function no longer takes `&mut self` and returns `SocketAddr`.
  The change in mutability is done to express that multiple `accept` calls can
  happen concurrently.
* For convenience the iterator does not yield the `SocketAddr` from `accept`.

The `TcpListener` type will also adhere to `Send` and `Sync`.

#### UDP
[UDP]: #udp

The UDP infrastructure will receive a similar face-lift as the TCP
infrastructure will:

```rust
impl UdpSocket {
    fn bind<A: ToSocketAddrs>(addr: &A) -> io::Result<UdpSocket>;
    fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)>;
    fn send_to<A: ToSocketAddrs>(&self, buf: &[u8], addr: &A) -> io::Result<usize>;
    fn local_addr(&self) -> io::Result<SocketAddr>;
    fn try_clone(&self) -> io::Result<UdpSocket>;
}

#[cfg(unix)]    impl AsRawFd for UdpSocket { ... }
#[cfg(windows)] impl AsRawSocket for UdpSocket { ... }
```

Some important points of note are:

* The `send` and `recv` function take `&self` instead of `&mut self` to indicate
  that they may be called safely in concurrent contexts.
* All configuration options such as `multicast` and `ttl` are left as
  `#[unstable]` for now.
* All timeout support is removed. This may come back in the form of `setsockopt`
  (as with TCP streams) or with a more general implementation of `select`.
* `clone` functionality has been replaced with `try_clone`.

The `UdpSocket` type will adhere to both `Send` and `Sync`.

#### Sockets
[Sockets]: #sockets

The current constructors for `TcpStream`, `TcpListener`, and `UdpSocket` are
largely "convenience constructors" as they do not expose the underlying details
that a socket can be configured before it is bound, connected, or listened on.
One of the more frequent configuration options is `SO_REUSEADDR` which is set by
default for `TcpListener` currently.

This RFC leaves it as an open question how best to implement this
pre-configuration. The constructors today will likely remain no matter what as
convenience constructors and a new structure would implement consuming methods
to transform itself to each of the various `TcpStream`, `TcpListener`, and
`UdpSocket`.

This RFC does, however, recommend not adding multiple constructors to the
various types to set various configuration options. This pattern is best
expressed via a flexible socket type to be added at a future date.

#### Addresses
[Addresses]: #addresses

For the current `addrinfo` module:

* The `get_host_addresses` should be renamed to `lookup_host`.
* All other contents should be removed.

For the current `ip` module:

* The `ToSocketAddr` trait should become `ToSocketAddrs`
* The default `to_socket_addr_all` method should be removed.

The following implementations of `ToSocketAddrs` will be available:

```rust
impl ToSocketAddrs for SocketAddr { ... }
impl ToSocketAddrs for SocketAddrV4 { ... }
impl ToSocketAddrs for SocketAddrV6 { ... }
impl ToSocketAddrs for (Ipv4Addr, u16) { ... }
impl ToSocketAddrs for (Ipv6Addr, u16) { ... }
impl ToSocketAddrs for (&str, u16) { ... }
impl ToSocketAddrs for str { ... }
impl<T: ToSocketAddrs> ToSocketAddrs for &T { ... }
```

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

* Replace `ToCStr` bounds with `AsOsStr`.
* Replace `env_set_all` with `env_clear`
* Rename `cwd` to `current_dir`, take `AsPath`.
* Rename `spawn` to `run`
* Move `uid` and `gid` to an extension trait in `os::unix`
* Make `detached` take a `bool` (rather than always setting the
  command to detached mode).

The `stdin`, `stdout`, `stderr` methods will undergo a more
significant change. By default, the corresponding options will be
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
  but their types will change to newtyped readers and writers to hide the internal
  pipe infrastructure.
* The `kill` method is dropped, and `id` and `signal` will move to `os::platform` extension traits.
* `signal_exit`, `signal_kill`, `wait`, and `forget` will all stay as they are.
* `set_timeout` will be changed to use the `with_deadline` infrastructure.

There are also a few other related changes to the module:

* Rename `ProcessOutput` to `Output`
* Rename `ProcessExit` to `ExitStatus`, and hide its
  representation. Remove `matches_exit_status`, and add a `status`
  method yielding an `Option<i32>`
* Remove `MustDieSignal`, `PleaseExitSignal`.
* Remove `EnvMap` (which should never have been exposed).

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

> To be expanded in follow-up PRs.

## Wide string representation

(Text from @SimonSapin)

Rather than WTF-8, `OsStr` and `OsString` on Windows could use
potentially-ill-formed UTF-16 (a.k.a. "wide" strings), with a
different cost trade off.

Upside:
* No conversion between `OsStr` / `OsString` and OS calls.

Downsides:
* More expensive conversions between `OsStr` / `OsString` and `str` / `String`.
* These conversions have inconsistent performance characteristics between platforms. (Need to allocate on Windows, but not on Unix.)
* Some of them return `Cow`, which has some ergonomic hit.

The API (only parts that differ) could look like:

```rust
pub mod os_str {
    #[cfg(windows)]
    mod imp {
        type Buf = Vec<u16>;
        type Slice = [u16];
        ...
    }

    impl OsStr {
        pub fn from_str(&str) -> Cow<OsString, OsStr>;
        pub fn to_string(&self) -> Option<CowString>;
        pub fn to_string_lossy(&self) -> CowString;
    }

    #[cfg(windows)]
    pub mod windows{
        trait OsStringExt {
            fn from_wide_slice(&[u16]) -> Self;
            fn from_wide_vec(Vec<u16>) -> Self;
            fn into_wide_vec(self) -> Vec<u16>;
        }

        trait OsStrExt {
            fn from_wide_slice(&[u16]) -> Self;
            fn as_wide_slice(&self) -> &[u16];
        }
    }
}
```
