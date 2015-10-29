- Feature Name: N/A
- Start Date: 2015-09-21
- RFC PR: [rust-lang/rfcs#1291](https://github.com/rust-lang/rfcs/pull/1291)
- Rust Issue: N/A

# Summary

Promote the `libc` crate from the nursery into the `rust-lang` organization
after applying changes such as:

* Remove the internal organization of the crate in favor of just one flat
  namespace at the top of the crate.
* Set up a large number of CI builders to verify FFI bindings across many
  platforms in an automatic fashion.
* Define the scope of libc in terms of bindings it will provide for each
  platform.

# Motivation

The current `libc` crate is a bit of a mess unfortunately, having long since
departed from its original organization and scope of definition. As more
platforms have been added over time as well as more APIs in general, the
internal as well as external facing organization has become a bit muddled. Some
specific concerns related to organization are:

* There is a vast amount of duplication between platforms with some common
  definitions. For example all BSD-like platforms end up defining a similar set
  of networking struct constants with the same definitions, but duplicated in
  many locations.
* Some subset of `libc` is reexported at the top level via globs, but not all of
  `libc` is reexported in this fashion.
* When adding new APIs it's unclear what modules it should be placed into. It's
  not always the case that the API being added conforms to one of the existing
  standards that a module exist for and it's not always easy to consult the
  standard itself to see if the API is in the standard.
* Adding a new platform to liblibc largely entails just copying a huge amount of
  code from some previously similar platform and placing it at a new location in
  the file.

Additionally, on the technical and tooling side of things some concerns are:

* None of the FFI bindings in this module are verified in terms of testing.
  This means that they are both not automatically generated nor verified, and
  it's highly likely that there are a good number of mistakes throughout.
* It's very difficult to explore the documentation for libc on different
  platforms, but this is often one of the more important libraries to have
  documentation for across all platforms.

The purpose of this RFC is to largely propose a reorganization of the libc
crate, along with tweaks to some of the mundane details such as internal
organization, CI automation, how new additions are accepted, etc. These changes
should all help push `libc` to a more more robust position where it can be well
trusted across all platforms both now and into the future!

# Detailed design

All design can be previewed as part of an [in progress fork][libc] available on
GitHub. Additionally, all mentions of the `libc` crate in this RFC refer to the
external copy on crates.io, not the in-tree one in the `rust-lang/rust`
repository. No changes are being proposed (e.g. to stabilize) the in-tree copy.

[libc]: https://github.com/alexcrichton/libc

### What is this crate?

The primary purpose of this crate is to provide all of the definitions
necessary to easily interoperate with C code (or "C-like" code) on each of the
platforms that Rust supports. This includes type definitions (e.g. `c_int`),
constants (e.g. `EINVAL`) as well as function headers (e.g. `malloc`).

One question that typically comes up with this sort of purpose is whether the
crate is "cross platform" in the sense that it basically just works across the
platforms it supports. The `libc` crate, however, **is not intended to be cross
platform** but rather the opposite, an exact binding to the platform in
question. In essence, the `libc` crate is targeted as "replacement for
`#include` in Rust" for traditional system header files, but it makes no
effort to be help being portable by tweaking type definitions and signatures.

### The Home of `libc`

Currently this crate resides inside of the main `rust` repo of the `rust-lang`
organization, but this unfortunately somewhat hinders its development as it
takes awhile to land PRs and isn't quite as quick to release as external
repositories. As a result, this RFC proposes having the crate reside externally
in the `rust-lang` organization so additions can be made through PRs (tested
much more quickly).

The main repository will have a submodule pointing at the external repository to
continue building libstd.

### Public API

The `libc` crate will hide all internal organization of the crate from users of
the crate. All items will be reexported at the top level as part of a flat
namespace. This brings with it a number of benefits:

* The internal structure can evolve over time to better fit new platforms
  while being backwards compatible.
* This design matches what one would expect from C, where there's only a flat
  namespace available.
* Finding an API is quite easy as the answer is "it's always at the root".

A downside of this approach, however, is that the public API of `libc` will be
platform-specific (e.g. the set of symbols it exposes is different across
platforms), which isn't seen very commonly throughout the rest of the Rust
ecosystem today. This can be mitigated, however, by clearly indicating that this
is a platform specific library in the sense that it matches what you'd get if
you were writing C code across multiple platforms.

The API itself will include any number of definitions typically found in C
header files such as:

* C types, e.g. typedefs, primitive types, structs, etc.
* C constants, e.g. `#define` directives
* C statics
* C functions (their headers)
* C macros (exported as `#[inline]` functions in Rust)

As a technical detail, all `struct` types exposed in `libc` will be guaranteed
to implement the `Copy` and `Clone` traits. There will be an optional feature of
the library to implement `Debug` for all structs, but it will be turned off by
default.

### Changes from today

The [in progress][libc] implementation of this RFC has a number of API changes
and breakages from today's `libc` crate. Almost all of them are minor and
targeted at making bindings more correct in terms of faithfully representing the
underlying platforms.

There is, however, one large notable change from today's crate. The `size_t`,
`ssize_t`, `ptrdiff_t`, `intptr_t`, and `uintptr_t` types are all defined in
terms of `isize` and `usize` instead of known sizes. Brought up by @briansmith
on [#28096][isizeusize] this helps decrease the number of casts necessary in
normal code and matches the existing definitions on all platforms that `libc`
supports today. In the future if a platform is added where these type
definitions are not correct then new ones will simply be available for that
target platform (and casts will be necessary if targeting it).

[isizeusize]: https://github.com/rust-lang/rust/pull/28096

Note that part of this change depends upon removing the compiler's
lint-by-default about `isize` and `usize` being used in FFI definitions. This
lint is mostly a holdover from when the types were named `int` and `uint` and it
was easy to confuse them with C's `int` and `unsigned int` types.

The final change to the `libc` crate will be to bump its version to 1.0.0,
signifying that breakage has happened (a bump from 0.1.x) as well as having a
future-stable interface until 2.0.0.

### Scope of `libc`

The name "libc" is a little nebulous as to what it means across platforms. It
is clear, however, that this library must have a well defined scope up to which
it can expand to ensure that it doesn't start pulling in dozens of runtime
dependencies to bind all the system APIs that are found.

Unfortunately, however, this library also can't be "just libc" in the sense of
"just libc.so on Linux," for example, as this would omit common APIs like
pthreads and would also mean that pthreads would be included on platforms like
MUSL (where it is literally inside libc.a). Additionally, the purpose of libc
isn't to provide a cross platform API, so there isn't necessarily one true
definition in terms of sets of symbols that `libc` will export.

In order to have a well defined scope while satisfying these constraints, this
RFC proposes that this crate will have a scope that is defined separately for
each platform that it targets. The proposals are:

* Linux (and other unix-like platforms) - the libc, libm, librt, libdl, and
  libpthread libraries. Additional platforms can include libraries whose symbols
  are found in these libraries on Linux as well.
* OSX - the common library to link to on this platform is libSystem, but this
  transitively brings in quite a few dependencies, so this crate will refine
  what it depends upon from libSystem a little further, specifically:
  libsystem\_c, libsystem\_m, libsystem\_pthread, libsystem\_malloc and libdyld.
* Windows - the VS CRT libraries. This library is currently intended to be
  distinct from the `winapi` crate as well as bindings to common system DLLs
  found on Windows, so the current scope of `libc` will be pared back to just
  what the CRT contains. This notably means that a large amount of the current
  contents will be removed on Windows.

New platforms added to `libc` can decide the set of libraries `libc` will link
to and bind at that time.

### Internal structure

The primary change being made is that the crate will no longer be one large file
sprinkled with `#[cfg]` annotations. Instead, the crate will be split into a
tree of modules, and all modules will reexport the entire contents of their
children. Unlike most libraries, however, most modules in `libc` will be
hidden via `#[cfg]` at compile time. Each platform supported by `libc` will
correspond to a path from a leaf module to the root, picking up more
definitions, types, and constants as the tree is traversed upwards.

This organization provides a simple method of deduplication between platforms.
For example `libc::unix` contains functions found across all unix platforms
whereas `libc::unix::bsd` is a refinement saying that the APIs within are common
to only BSD-like platforms (these may or may not be present on non-BSD platforms
as well). The benefits of this structure are:

* For any particular platform, it's easy in the source to look up what its value
  is (simply trace the path from the leaf to the root, aka the filesystem
  structure, and the value can be found).
* When adding an API it's easy to know **where** the API should be added because
  each node in the module hierarchy corresponds clearly to some subset of
  platforms.
* Adding new platforms should be a relatively simple and confined operation. New
  leaves of the hierarchy would be created and some definitions upwards may be
  pushed to lower levels if APIs need to be changed or aren't present on the new
  platform. It should be easy to audit, however, that a new platform doesn't
  tamper with older ones.

### Testing

The current set of bindings in the `libc` crate suffer a drawback in that they
are not verified. This is often a pain point for new platforms where when
copying from an existing platform it's easy to forget to update a constant here
or there. This lack of testing leads to problems like a [wrong definition of
`ioctl`][ioctl] which in turn lead to [backwards compatibility
problems][backcompat] when the API is fixed.

[ioctl]: https://github.com/rust-lang/rust/pull/26809
[backcompat]: https://github.com/rust-lang/rust/pull/27762

In order to solve this problem altogether, the libc crate will be enhanced with
the ability to automatically test the FFI bindings it contains. As this crate
will begin to live in `rust-lang` instead of the `rust` repo itself, this means
it can leverage external CI systems like Travis CI and AppVeyor to perform these
tasks.

The [current implementation][ctest] of the binding testing verifies attributes
such as type size/alignment, struct field offset, struct field types, constant
values, function definitions, etc. Over time it can be enhanced with more
metrics and properties to test.

[ctest]: https://github.com/alexcrichton/ctest

In theory adding a new platform to `libc` will be blocked until automation can
be set up to ensure that the bindings are correct, but it is unfortunately not
easy to add this form of automation for all platforms, so this will not be a
requirement (beyond "tier 1 platforms"). There is currently automation for the
following targets, however, through Travis and AppVeyor:

* `{i686,x86_64}-pc-windows-{msvc,gnu}`
* `{i686,x86_64,mips,aarch64}-unknown-linux-gnu`
* `x86_64-unknown-linux-musl`
* `arm-unknown-linux-gnueabihf`
* `arm-linux-androideabi`
* `{i686,x86_64}-apple-{darwin,ios}`

# Drawbacks

### Loss of module organization

The loss of an internal organization structure can be seen as a drawback of this
design. While perhaps not precisely true today, the principle of the structure
was that it is easy to constrain yourself to a particular C standard or subset
of C to in theory write "more portable programs by default" by only using the
contents of the respective module. Unfortunately in practice this does not seem
to be that much in use, and it's also not clear whether this can be expressed
through simply headers in `libc`. For example many platforms will have slight
tweaks to common structures, definitions, or types in terms of signedness or
value, so even if you were restricted to a particular subset it's not clear that
a program would automatically be more portable.

That being said, it would still be useful to have these abstractions to *some
degree*, but the filp side is that it's easy to build this sort of layer on top
of `libc` as designed here externally on crates.io. For example `extern crate
posix` could just depend on `libc` and reexport all the contents for the
POSIX standard, perhaps with tweaked signatures here and there to work better
across platforms.

### Loss of Windows bindings

By only exposing the CRT functions on Windows, the contents of `libc` will be
quite trimmed down which means when accessing similar functions like `send` or
`connect` crates will be required to link to two libraries at least.

This is also a bit of a maintenance burden on the standard library itself as it
means that all the bindings it uses must move to `src/libstd/sys/windows/c.rs`
in the immedidate future.

# Alternatives

* Instead of *only* exporting a flat namespace the `libc` crate could optionally
  also do what it does today with respect to reexporting modules corresponding
  to various C standards. The downside to this, unfortunately, is that it's
  unclear how much portability using these standards actually buys you.

* The crate could be split up into multiple crates which represent an exact
  correspondance to system libraries, but this has the downside of using common
  functions available on both OSX and Linux would require at least two `extern
  crate` directives and dependencies.

# Unresolved questions

* The only platforms without automation currently are the BSD-like platforms
  (e.g. FreeBSD, OpenBSD, Bitrig, DragonFly, etc), but if it were possible to
  set up automation for these then it would be plausible to actually require
  automation for any new platform. It is possible to do this?

* What is the relation between `std::os::*::raw` and `libc`? Given that the
  standard library will probably always depend on an in-tree copy of the `libc`
  crate, should `libc` define its own in this case, have the standard library
  reexport, and then the out-of-tree `libc` reexports the standard library?

* Should Windows be supported to a greater degree in `libc`? Should this crate
  and `winapi` have a closer relationship?
