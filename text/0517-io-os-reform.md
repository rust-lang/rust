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
    * [Revising `Reader` and `Writer`] (stub)
    * [String handling]
        * [Key observations]
        * [The design: `os_str`]
        * [The future]
    * [Deadlines] (stub)
    * [Splitting streams and cancellation] (stub)
    * [Modules]
        * [core::io] (stub)
        * [The std::io facade] (stub)
        * [std::env]
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

> To be added in a follow-up PR.

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

> To be added in a follow-up PR.

### The `std::io` facade
[The std::io facade]: #the-stdio-facade

> To be added in a follow-up PR.

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

* `join_paths`: take an `IntoIterator<T>` where `T: IntoOsStrBuf`, yield a `Result<OsStrBuf, JoinPathsError>`.
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
