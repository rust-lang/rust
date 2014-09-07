- Start Date: 2014-04-08
- RFC PR: [rust-lang/rfcs#40](https://github.com/rust-lang/rfcs/pull/40)
- Rust Issue: [rust-lang/rust#13851](https://github.com/rust-lang/rust/issues/13851)

# Summary

Split the current libstd into component libraries, rebuild libstd as a facade in
front of these component libraries.

# Motivation

Rust as a language is ideal for usage in constrained contexts such as embedding
in applications, running on bare metal hardware, and building kernels. The
standard library, however, is not quite as portable as the language itself yet.
The standard library should be as usable as it can be in as many contexts as
possible, without compromising its usability in any context.

This RFC is meant to expand the usability of the standard library into these
domains where it does not currently operate easily

# Detailed design

In summary, the following libraries would make up part of the standard
distribution. Each library listed after the colon are the dependent libraries.

* libmini
* liblibc
* liballoc: libmini liblibc
* libcollections: libmini liballoc
* libtext: libmini liballoc libcollections
* librustrt: libmini liballoc liblibc
* libsync: libmini liballoc liblibc librustrt
* libstd: everything above

### `libmini`

> **Note**: The name `libmini` warrants bikeshedding. Please consider it a
>           placeholder for the name of this library.

This library is meant to be the core component of *all rust programs in
existence*. This library has very few external dependencies, and is entirely
self contained.

Current modules in `std` which would make up libmini would include the list
below. This list was put together by actually stripping down libstd to these
modules, so it is known that it is possible for libmini to compile with these
modules.

* `atomics`
* `bool`
* `cast`
* `char`
* `clone`
* `cmp`
* `container`
* `default`
* `finally`
* `fmt`
* `intrinsics`
* `io`, stripped down to its core
* `iter`
* `kinds`
* `mem`
* `num` (and related modules), no float support
* `ops`
* `option`
* `ptr`
* `raw`
* `result`
* `slice`, but without any `~[T]` methods
* `tuple`
* `ty`
* `unit`

This list may be a bit surprising, and it's makeup is discussed below. Note that
this makeup is selected specifically to eliminate the need for the dreaded "one
off extension trait". This pattern, while possible, is currently viewed as
subpar due to reduced documentation benefit and sharding implementation across
many locations.

#### Strings

In a post-DST world, the string type will actually be a library-defined type,
`Str` (or similarly named). Strings will no longer be a lanuage feature or a
language-defined type. This implies that any methods on strings must be in the
same crate that defined the `Str` type, or done through extension traits.

In the spirit of reducing extension traits, the `Str` type and module were left
out of libmini. It's impossible for libmini to support all methods of `Str`, so
it was entirely removed.

This decision does have ramifications on the implementation of `libmini`.

* String literals are an open question. In theory, making a string literal would
  require the `Str` lang item to be present, but is not present in libmini. That
  being said, libmini would certainly create many literal strings (for error
  messages and such). This may be adequately circumvented by having literal
  strings create a value of type `&'static [u8]` if the string lang item is not
  present. While difficult to work with, this may get us 90% of the way there.

* The `fmt` module must be tweaked for the removal of strings.
  The only major user-facing detail is that the `pad` function on `Formatter`
  would take a byte-slice and a character length, and then not handle the
  precision (which truncates the byte slice with a number of characters). This
  may be overcome by possibly having an extension trait could be added for a
  `Formatter` adding a real `pad` function that takes strings, or just removing
  the function altogether in favor of `str.fmt(formatter)`.

* The `IoError` type suffers from the removal of strings. Currently, this type
  is inhabited with three fields, an enum, a static description string, and an
  optionally allocated detail string. Removal of strings would imply the
  `IoError` type would be just the enum itself. This may be an acceptable
  compromise to make, defining the `IoError` type upstream and providing easy
  constructors from the enum to the struct. Additionally, the `OtherIoError`
  enum variant would be extended with an `i32` payload representing the error
  code (if it came from the OS).

* The `ascii` module is omitted, but it would likely be defined in the crate
  that defines `Str`.

#### Formatting

While not often thought of as "ultra-core" functionality, this module may be
necessary because printing information about types is a fundamental problem that
normally requires no dependencies.

Inclusion of this module is the reason why I/O is included in the module as well
(or at least a few traits), but the module can otherwise be included with little
to no overhead required in terms of dependencies.

Neither `print!` nor `format!` macros to be a part of this library, but the
`write!` macro would be present.

#### I/O

The primary reason for defining the `io` module in the libmini crate would be to
implement the `fmt` module. The ramification of removing strings was previously
discussed for `IoError`, but there are further modifications that would be
required for the `io` module to exist in libmini:

* The `Buffer`, `Listener`, `Seek`, and `Acceptor` traits would all be defined
  upstream instead of in libmini. Very little in libstd uses these traits, and
  nothing in libmini requires them. They are of questionable utility when
  considering their applicability to all rust code in existence.

* Some extension methods on the `Reader` and `Writer` traits would need to be
  removed.  Methods such as `push_exact`, `read_exact`, `read_to_end`,
  `write_line`, etc., all require owned vectors or similar unimplemented runtime
  requirements. These can likely be moved to extension traits upstream defined
  for all readers and writers. Note that this does not apply to the integral
  reading and writing methods. These are occasionally overwritten for
  performance, but removal of some extension methods would strongly suggest to
  me that these methods should be removed. Regardless, the remaining methods
  could live in essentially any location.

#### Slices

The only method lost on mutable slices would currently be the sorting method.
This can be circumvented by implementing a sorting algorithm that doesn't
require allocating a temporary buffer. If intensive use of a sorting algorithm
is required, Rust can provide a `libsort` crate with a variety of sorting
algorithms apart from the default sorting algorithm.

#### FromStr

This trait and module are left out because strings are left out. All types in
libmini can have their implemention of FromStr in the crate which implements
strings

#### Floats

This current design excludes floats entirely from libmini (implementations of
traits and such). This is another questionable decision, but the current
implementation of floats heavily leans on functions defined in libm, so it is
unacceptable for these functions to exist in libmini.

Either libstd or a libfloat crate will define floating point traits and such.

#### Failure

It is unacceptable for `Option` to reside outside of libmini, but it is also
also unacceptable for `unwrap` to live outside of the `Option` type.
Consequently, this means that it must be possible for `libmini` to fail.

While impossible for libmini to *define* failure, it should simply be able to
*declare* failure. While currently not possible today, this extension to the
language is possible through "weak lang items".

Implementation-wise, the failure lang item would have a predefined symbol at
which it is defined, and libraries which *declare* but to not *define* failure
are required to only exist in the rlib format. This implies that libmini can
*only* be built as an rlib. Note that today's linkage rules do not allow for
this (because building a dylib with rlib dependencies is not possible), but the
rules could be tweaked to allow for this use case.

tl;dr; The implementation of libmini can use failure, but it does not define
failure. All usage of libmini would require an implementation of failure
somewhere.

### `liblibc`

This library will exist to provide bindings to libc. This will be a highly
platform-specific library, containing an entirely separate api depending on
which platform it's being built for.

This crate will be used to provide bindings to the C language in all forms, and
would itself essentially be a giant metadata blob. It conceptually represents
the inclusion of all C header files.

Note that the funny name of the library is to allow `extern crate libc;` to be
the form of declaration rather than `extern crate c;` which is consider to be
too short for its own good.

Note that this crate can only exist in rlib or dylib form.

### `liballoc`

> **Note**: This name `liballoc` is questionable, please consider it a
>           placeholder.

This library would define the allocator traits as well as bind to libc
malloc/free (or jemalloc if we decide to include it again). This crate would
depend on liblibc and libmini.

Pointers such as `~` and Rc would move into this crate using the default
allocator. The current Gc pointers would move to libgc if possible, or otherwise
librustrt for now (they're feature gated currently, not super pressing).

Primarily, this library assumes that an allocation failure should trigger a
failure. This makes the library not suitable for use in a kernel, but it is
suitable essentially everywhere else.

With today's libstd, this crate would likely mostly be made up by the
`global_heap` module. Its purpose is to define the allocation lang items
required by the compiler.

Note that this crate can only exist in rlib form.

### `libcollections`

This crate would *not* depend on libstd, it would only depend on liballoc and
libmini. These two foundational crates should provide all that is necessary to
provide a robust set of containers (what you would expect today). Each container
would likely have an allocator parameter, and the default would be the default
allocator provided by liballoc.

When using the containers from libcollections, it is implicitly assumed that all
allocation succeeds, and this will be reflected in the api of each collection.

The contents of this crate would be the entirety of `libcollections` as it is
today, as well as the `vec` module from the standard library. This would also
implement any relevant traits necessary for `~[T]`.

Note that this crate can only exist in rlib form.

### `libtext`

This crate would define all functionality in rust related to strings. This would
contain the definition of the `Str` type, as well as implementations of the
relevant traits from `libmini` for the string type.

The crucial assumption of this crate is that allocation does not fail, and the
rest of the string functionality could be built on top of this. Note that this
crate will depend on `libcollections` for the `Vec` type as the underlying
building block for string buffers and the string type.

This crate would be composed of the `str`, `ascii`, and `unicode` modules which
live in libstd today, but would allow for the extension of other text-related
functionality.

### `librustrt`

This library would be the crate where the `rt` module is almost entirely
implemented. It will assume that allocation succeeds, and it will assume a libc
implementation to run on.

The current libstd modules which would be implemented as part of this crate
would be:

* `rt`
* `task`
* `local_data`

Note that `comm` is *not* on this list. This crate will additionally define
failure (as unwinding for each task). This crate can exist in both rlib and
dylib form.

### `libsync`

This library will largely remain what it is today, with the exception that the
`comm` implementation would move into this crate. The purpose of doing so would
be to consolidate all concurrency-related primitives in this crate, leaving none
out.

This crate would depend on the runtime for task management (scheduling and
descheduling).

### The `libstd` facade

A new standard library would be created that would primarily be a facade which
would expose the underlying crates as a stable API. This library would depend on
all of the above libraries, and would predominately be a grouping of `pub use`
statements.

This library would also be the library to contain the prelude which would
include types from the previous crates. All remaining functionality of the
standard library would be filled in as part of this crate.

Note that all rust programs will by default link to `libstd`, and hence will
transitively link to all of the upstream crates mentioned above. Many more apis
will be exposed through `libstd` directly, however, such as `HashMap`, `Arc`,
etc.

The exact details of the makeup of this crate will change over time, but it can
be considered as "the current libstd plus more", and this crate will be the
source of the "batteries included" aspect of the rust standard library. The API
(reexported paths) of the standard library would not change over time. Once a
path is reexported and a release is made, all the path will be forced to remain
constant over time.

One of the primary reasons for this facade is to provide freedom to restructure
the underlying crates. Once a facade is established, it is the only stable API.
The actual structure and makeup of all the above crates will be fluid until an
acceptable design is settled on. Note that this fluidity does not apply to
libstd, only to the structure of the underlying crates.

### Updates to rustdoc

With today's incarnation of rustdoc, the documentation for this libstd facade
would not be as high quality as it is today. The facade would just provide
hyperlinks back to the original crates, which would have reduced quantities of
documentation in terms of navigation, implemented traits, etc. Additionally,
these reexports are meant to be implementation details, not facets of the api.
For this reason, rustdoc would have to change in how it renders documentation
for libstd.

First, rustdoc would consider a cross-crate reexport as inlining of the
documentation (similar to how it inlines reexports of private types). This would
allow all documentation in libstd to remain in the same location (even the same
urls!). This would likely require extensive changes to rustdoc for when entire
module trees are reexported.

Secondly, rustdoc will have to be modified to collect implementors of reexported
traits all in one location. When libstd reexports trait X, rustdoc will have to
search libstd and all its dependencies for implementors of X, listing them out
explicitly.

These changes to rustdoc should place it in a much more presentable space, but
it is an open question to what degree these modifications will suffice and how
much further rustdoc will have to change.

### Remaining crates

There are many more crates in the standard distribution of rust, all of which
currently depend on libstd. These crates would continue to depend on libstd as
most rust libraries would.

A new effort would likely arise to reduce dependence on the standard library by
cutting down to the core dependencies (if necessary). For example, the
`libnative` crate currently depend on `libstd`, but it in theory doesn't need to
depend on much other than `librustrt` and `liblibc`. By cutting out
dependencies, new use cases will likely arise for these crates.

Crates outside of the standard distribution of rust will like to link to the
above crates as well (and specifically not libstd). For example, crates which
only depend on libmini are likely candidates for being used in kernels, whereas
crates only depending on liballoc are good candidates for being embedded into
other languages. Having a clear delineation for the usability of a crate in
various environments seems beneficial.

# Alternatives

* There are many alternatives to the above sharding of libstd and its dependent
  crates. The one that is most rigid is likely libmini, but the contents of all
  other crates are fairly fluid and able to shift around. To this degree, there
  are quite a few alternatives in how the remaining crates are organized. The
  ordering proposed is simply one of many.

* Compilation profiles. Instead of using crate dependencies to encode where a
  crate can be used, crates could instead be composed of `cfg(foo)` attributes.
  In theory, there would be one `libstd` crate (in terms of source code), and
  this crate could be compiled with flags such as `--cfg libc`, `--cfg malloc`,
  etc. This route has may have the problem of "multiple standard libraries"
  in that code compatible with the "libc libstd" is not necessarily compatible
  with the "no libc libstd". Asserting that a crate is compatible with multiple
  profiles would involve requiring multiple compliations.

* Removing libstd entirely. If the standard library is simply a facade, the
  compiler could theoretically only inject a select number of crates into the
  prelude, or possibly even omit the prelude altogether. This works towards
  elimination the question of "does this belong in libstd", but it would
  possibly be difficult to juggle the large number of crates to choose from
  where one could otherwise just look at libstd.

# Unresolved questions

* Compile times. It's possible that having so many upstream crates for each rust
  crate will increase compile times through reading metadata and invoking the
  system linker. Would sharding crates still be worth it? Could possible
  problems that arise be overcome? Would extra monomorphization in all these
  crates end up causing more binary bloat?

* Binary bloat. Another possible side effect of having many upstream crates
  would be increasing binary bloat of each rust program. Our current linkage
  model means that if you use anything from a crate that you get *everything* in
  that crate (in terms of object code). It is unknown to what degree this will
  become a concern, and to what degree it can be overcome.

* Should floats be left out of libmini? This is largely a question of how much
  runtime support is required for floating point operations. Ideally
  functionality such as formatting a float would live in libmini, whereas
  trigonometric functions would live in an external crate with a dependence on
  libm.

* Is it acceptable for strings to be left out of libmini? Many common operations
  on strings don't require allocation. This is currently done out of necessity
  of having to define the Str type elsewhere, but this may be seen as too
  limiting for the scope of libmini.

* Does liblibc belong so low in the dependency tree? In the proposed design,
  only the libmini crate doesn't depend on liblibc. Crates such as libtext and
  libcollections, however, arguably have no dependence on libc itself, they
  simply require some form of allocator. Answering this question would be
  figuring how how to break liballoc's dependency on liblibc, but it's an open
  question as to whether this is worth it or not.

* Reexporting macros. Currently the standard library defines a number of useful
  macros which are used throughout the implementation of libstd. There is no way
  to reexport a macro, so multiple implementations of the same macro would be
  required for the core libraries to all use the same macro. Is there a better
  solution to this situation? How much of an impact does this have?
