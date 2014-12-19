- Start Date: 2014-11-12
- RFC PR: [rust-lang/rfcs#474](https://github.com/rust-lang/rfcs/pull/474)
- Rust Issue: [rust-lang/rust#20034](https://github.com/rust-lang/rust/issues/20034)

# Summary

This RFC reforms the design of the `std::path` module in preparation for API
stabilization. The path API must deal with many competing demands, and the
current design handles many of them, but suffers from some significant problems
given in "Motivation" below. The RFC proposes a redesign modeled loosely on the
current API that addresses these problems while maintaining the advantages of
the current design.

# Motivation

The design of a path abstraction is surprisingly hard. Paths work radically
differently on different platforms, so providing a cross-platform abstraction is
challenging. On some platforms, paths are not required to be in Unicode, posing
ergonomic and semantic difficulties for a Rust API. These difficulties are
compounded if one also tries to provide efficient path manipulation that does
not, for example, require extraneous copying. And, of course, the API should be
easy and pleasant to use.

The current `std::path` module makes a strong effort to balance these design
constraints, but over time a few key shortcomings have emerged.

## Semantic problems

Most importantly, the current `std::path` module makes some semantic assumptions
about paths that have turned out to be incorrect.

### Normalization

Paths in `std::path` are always *normalized*, meaning that `a/../b` is treated
like `b` (among other things). Unfortunately, this kind of normalization changes
the meaning of paths when symbolic links are present: if `a` is a symbolic link,
then the relative paths `a/../b` and `b` may refer to completely different
locations. See [this issue](https://github.com/rust-lang/rust/issues/14028) for
more detail.

For this reason, most path libraries do *not* perform full normalization of
paths, though they may normalize paths like `a/./b` to `a/b`. Instead, they
offer (1) methods to optionally normalize and (2) methods to normalize based on
the contents of the underlying file system.

Since our current normalization scheme can silently and incorrectly alter the
meaning of paths, it needs to be changed.

### Unicode and Windows

In the original `std::path` design, it was assumed that all paths on Windows
were Unicode. However, it
[turns out](https://github.com/rust-lang/rust/issues/12056) that the Windows
filesystem APIs actually work with [UCS-2](http://en.wikipedia.org/wiki/UTF-16),
which roughly means that they accept arbitrary sequences of `u16` values but
interpret them as UTF-16 when it is valid to do so.

The current `std::path` implementation is built around the assumption that
Windows paths can be represented as Rust string slices, and will need to be
substantially revised.

## Ergonomic problems

Because paths in general are not in Unicode, the `std::path` module cannot rely on
an internal string or string slice representation. That in turn causes trouble
for methods like `dirname` that are intended to extract a subcomponent of a path
-- what should it return?

There are basically three possible options, and today's `std::path` module
chooses *all* of them:

* Yield a byte sequence: `dirname` yields an `&[u8]`
* Yield a string slice, accounting for potential non-UTF-8 values: `dirname_str`
  yields an `Option<&str>`
* Yield another path: `dir_path` yields a `Path`

This redundancy is present for most of the decomposition methods. The saving
grace is that, in general, path methods consume `BytesContainer` values, so one
can use the `&[u8]` variant but continue to work with other path methods. But in
general `&[u8]` values are not ergonomic to work with, and the explosion in
methods makes the module more (superficially) complex than one might expect.

You might be tempted to provide only the third option, but `Path` values are
*owned* and *mutable*, so that would imply cloning on every decomposition
operation. For applications like Cargo that work heavily with paths, this would
be an unfortunate (and seemingly unnecessary) overhead.

## Organizational problems

Finally, the `std::path` module presents a somewhat complex API organization:

* The `Path` type is a direct alias of a platform-specific path type.
* The `GenericPath` trait provides most of the common API expected on both platforms.
* The `GenericPathUnsafe` trait provides a few unsafe/unchecked functions for
  performance reasons.
* The `posix` and `windows` submodules provide their own `Path` types and a
  handful of platform-specific functionality (in particular, `windows` provides
  support for working with volumes and "verbatim" paths prefixed with `\\?\`)

This organization needs to be updated to match current conventions and
simplified if possible.

One thing to note: with the current organization, it is possible to work with
non-native paths, which can sometimes be useful for interoperation. The new
design should retain this functionality.

# Detailed design

Note: this design is influenced by the
[Boost filesystem library](www.boost.org/doc/libs/1_57_0/libs/filesystem/doc/reference.html)
and [Scheme48](http://s48.org/1.8/manual/manual-Z-H-6.html#node_sec_5.15) and
[Racket's](http://plt.eecs.northwestern.edu/snapshots/current/doc/reference/windowspaths.html#%28part._windowspathrep%29)
approach to encoding issues on windows.

## Overview

The basic design uses DST to follow the same pattern as `Vec<T>/[T]` and
`String/str`: there is a `PathBuf` type for owned, mutable paths and an unsized
`Path` type for slices. The various "decomposition" methods for extracting
components of a path all return slices, and `PathBuf` itself derefs to `Path`.

The result is an API that is both efficient and ergonomic: there is no need to
allocate/copy when decomposing a path, but there is also no need to provide
multiple variants of methods to extract bytes versus Unicode strings. For
example, the `Path` slice type provides a *single* method for converting to a
`str` slice (when applicable).

A key aspect of the design is that there is no internal normalization of paths
at all. Aside from solving the symbolic link problem, this choice also has
useful ramifications for the rest of the API, described below.

The proposed API deals with the other problems mentioned above, and also brings
the module in line with current Rust patterns and conventions. These details
will be discussed after getting a first look at the core API.

## The cross-platform API

The proposed core, cross-platform API provided by the new `std::path` is as follows:

```rust
// A sized, owned type akin to String:
pub struct PathBuf { .. }

// An unsized slice type akin to str:
pub struct Path { .. }

// Some ergonomics and generics, following the pattern in String/str and Vec<T>/[T]
impl Deref<Path> for PathBuf { ... }
impl BorrowFrom<PathBuf> for Path { ... }

// A replacement for BytesContainer; used to cut down on explicit coercions
pub trait AsPath for Sized? {
    fn as_path(&self) -> &Path;
}

impl<Sized? P> PathBuf where P: AsPath {
    pub fn new<T: IntoString>(path: T) -> PathBuf;

    pub fn push(&mut self, path: &P);
    pub fn pop(&mut self) -> bool;

    pub fn set_file_name(&mut self, file_name: &P);
    pub fn set_extension(&mut self, extension: &P);
}

// These will ultimately replace the need for `push_many`
impl<Sized? P> FromIterator<P> for PathBuf where P: AsPath { .. }
impl<Sized? P> Extend<P> for PathBuf where P: AsPath { .. }

impl<Sized? P> Path where P: AsPath {
    pub fn new(path: &str) -> &Path;

    pub fn as_str(&self) -> Option<&str>
    pub fn to_str_lossy(&self) -> Cow<String, str>; // Cow will replace MaybeOwned
    pub fn to_owned(&self) -> PathBuf;

    // iterate over the components of a path
    pub fn iter(&self) -> Iter;

    pub fn is_absolute(&self) -> bool;
    pub fn is_relative(&self) -> bool;
    pub fn is_ancestor_of(&self, other: &P) -> bool;

    pub fn path_relative_from(&self, base: &P) -> Option<PathBuf>;
    pub fn starts_with(&self, base: &P) -> bool;
    pub fn ends_with(&self, child: &P) -> bool;

    // The "root" part of the path, if absolute
    pub fn root_path(&self) -> Option<&Path>;

    // The "non-root" part of the path
    pub fn relative_path(&self) -> &Path;

    // The "directory" portion of the path
    pub fn dir_path(&self) -> &Path;

    pub fn file_name(&self) -> Option<&Path>;
    pub fn file_stem(&self) -> Option<&Path>;
    pub fn extension(&self) -> Option<&Path>;

    pub fn join(&self, path: &P) -> PathBuf;

    pub fn with_file_name(&self, file_name: &P) -> PathBuf;
    pub fn with_extension(&self, extension: &P) -> PathBuf;
}

pub struct Iter<'a> { .. }

impl<'a> Iterator<&'a Path> for Iter<'a> { .. }

pub const SEP: char = ..
pub const ALT_SEPS: &'static [char] = ..

pub fn is_separator(c: char) -> bool { .. }
```

There is plenty of overlap with today's API, and the methods being retained here
largely have the same semantics.

But there are also a few potentially surprising aspects of this design that merit
comment:

* **Why does `PathBuf::new` take `IntoString`?** It needs an owned buffer
  internally, and taking a string means that Unicode input is guaranteed, which
  works on all platforms. (In general, the assumption is that non-Unicode paths
  are most commonly produced by *reading* a path from the filesystem, rather
  than creating now ones. As we'll see below, there are *platform-specific* ways
  to crate non-Unicode paths.)

* **Why no `Path::as_bytes` method?** There is no cross-platform way to expose
  paths directly in terms of byte sequences, because each platform extends
  beyond Unicode in its own way. In particular, Unix platforms accept arbitrary
  u8 sequences, while Windows accepts arbitrary *u16* sequences (both modulo
  disallowing interior 0s). The u16 sequences provided by Windows do not have a
  canonical encoding as bytes; this RFC proposed to use
  [WTF-8](http://simonsapin.github.io/wtf-8/) (see below), but does not reveal
  that choice.

* **What about interior nulls?** Currently various Rust system APIs will panic
  when given strings containing interior null values because, while these are
  valid UTF-8, it is not possible to send them as-is to C APIs that expect
  null-terminated strings. The API here follows the same approach, panicking if
  given a path with an interior null.

* **Why do `file_name` and `extension` operations work with `Path` rather than
  some other type?** In particular, it may seem strange to view an extension as
  a path. But doing so allows us to not reveal platform differences about the
  various character sets used in paths. By and large, extensions in practice will
  be valid Unicode, so the various methods going to and from `str` will
  suffice. But as with paths in general, there are platform-specific ways of
  working with non-Unicode data, explained below.

* **Where did `push_many` and friends go?** They're replaced by implementing
  `FromIterator` and `Extend`, following a similar pattern with the `Vec`
  type. (Some work will be needed to retain full efficiency when doing so.)

* **How does `Path::new` work?** The ability to directly get a `&Path` from an
  `&str` (i.e., with no allocation or other work) is a key part of the
  representation choices, which are described below.

* **Where is the `normalize` method?** Since the path type no longer internally
  normalizes, it may be useful to explicitly request normalization. This can be
  done by writing `let normalized: PathBuf = p.iter().collect()` for a path `p`,
  because the iterator performs some on-the-fly normalization (see
  below). **NOTE* this normalization does *not* include removing `..`, for the
  reasons explained at the beginning of the RFC.

* **What does the iterator yield?** Unlike today's `components`, the `iter`
  method here will begin with `root_path` if there is one. Thus, `a/b/c` will
  yield `a`, `b` and `c`, while `/a/b/c` will yield `/`, `a`, `b` and `c`.

## Important semantic rules

The path API is designed to satisfy several semantic rules described below.
**Note that `==` here is *lazily* normalizing**, treating `./b` as `b` and
`a//b` as `a/b`; see the next section for more details.

Suppose `p` is some `&Path` and `dot == Path::new(".")`:

```rust
p == p.join(dot)
p == dot.join(p)

p == p.root_path().unwrap_or(dot)
      .join(p.relative_path())

p.relative_path() == match p.root_path() {
    None => p,
    Some(root) => p.path_relative_from(root).unwrap()
}

p == p.dir_path()
      .join(p.file_name().unwrap_or(dot))

p == p.iter().collect()

p == match p.file_name() {
    None => p,
    Some(name) => p.with_file_name(name)
}

p == match p.extension() {
    None => p,
    Some(ext) => p.with_extension(ext)
}

p == match (p.file_stem(), p.extension()) {
    (Some(stem), Some(ext)) => p.with_file_name(name).with_extension(ext),
    _ => p
}
```

## Representation choices, Unicode, and normalization

A lot of the design in this RFC depends on a key property: both Unix and Windows
paths can be easily represented as a flat byte sequence "compatible" with
UTF-8. For Unix platforms, this is trivial: they accept any byte sequence, and
will generally interpret the byte sequences as UTF-8 when valid to do so. For
Windows, this representation involves a clever hack -- proposed formally as
[WTF-8](http://simonsapin.github.io/wtf-8/) -- that encodes its native UCS-2 in
a generalization of UTF-8. This RFC will not go into the details of that hack;
please read Simon's excellent writeup if you're interested.

The upshot of all of this is that we can uniformly represent path slices as
newtyped byte slices, and any UTF-8 encoded data will "do the right thing" on
all platforms.

Furthermore, by not doing any internal, up-front normalization, it's possible to
provide a `Path::new` that goes from `&str` to `&Path` with no intermediate
allocation or validation. In the common case that you're working with Rust
strings to construct paths, there is zero overhead. It also means that
`Path::new(some_str).as_str = Some(some_str)`.

The main downside of this choice is that some of the path functionality must
cope with non-normalized paths. So, for example, the iterator must skip `.` path
components (unless it is the entire path), and similarly for methods like
`pop`. In general, methods that yield new path slices are expected to work as if:

* `./b` is just `b`
* `a//b` is just `a/b`

and comparisons between paths should also behave as if the paths had been
normalized in this way.

## Organization and platform-specific APIs

Finally, the proposed API is organized as `std::path` with `unix` and `windows`
submodules, as today. However, there is no `GenericPath` or `GenericPathUnsafe`;
instead, the API given above is implemented as a trivial wrapper around path
implementations provided by either the `unix` or the `windows` submodule (based
on `#[cfg]`). In other words:

* `std::path::windows::Path` works with Windows-style paths
* `std::path::unix::Path` works with Unix-style paths
* `std::path::Path` is a thin newtype wrapper around the current platform's path implementation

This organization makes it possible to manipulate foreign paths by working with
the appropriate submodule.

In addition, each submodule defines some extension traits, explained below, that
supplement the path API with functionality relevant to its variant of path.

But what if you're writing a platform-specific application and wish to use the
extended functionality directly on `std::path::Path`? In this case, you will be
able to import the appropriate extension trait via `os::unix` or `os::windows`,
depending on your platform. This is part of a new, general strategy for
explicitly "opting-in" to platform-specific features by importing from
`os::some_platform` (where the `some_platform` submodule is available only on
that platform.)

### Unix

On Unix platforms, the only additional functionality is to let you work directly
with the underlying byte representation of various path types:

```rust
pub trait UnixPathBufExt {
    fn from_vec(path: Vec<u8>) -> Self;
    fn into_vec(self) -> Vec<u8>;
}

pub trait UnixPathExt {
    fn from_bytes(path: &[u8]) -> &Self;
    fn as_bytes(&self) -> &[u8];
}
```

This is acceptable because the platform supports arbitrary byte sequences
(usually interpreted as UTF-8).

### Windows

On Windows, the additional APIs allow you to convert to/from UCS-2 (roughly,
arbitrary `u16` sequences interpreted as UTF-16 when applicable); because the
name "UCS-2" does not have a clear meaning, these APIs use `u16_slice` and will
be carefully documented. They also provide the remaining Windows-specific path
decomposition functionality that today's path module supports.

```rust
pub trait WindowsPathBufExt {
    fn from_u16_slice(path: &[u16]) -> Self;
    fn make_non_verbatim(&mut self) -> bool;
}

pub trait WindowsPathExt {
    fn is_cwd_relative(&self) -> bool;
    fn is_vol_relative(&self) -> bool;
    fn is_verbatim(&self) -> bool;
    fn prefix(&self) -> PathPrefix;
    fn to_u16_slice(&self) -> Vec<u16>;
}

enum PathPrefix<'a> {
    Verbatim(&'a Path),
    VerbatimUNC(&'a Path, &'a Path),
    VerbatimDisk(&'a Path),
    DeviceNS(&'a Path),
    UNC(&'a Path, &'a Path),
    Disk(&'a Path),
}
```

# Drawbacks

The DST/slice approach is conceptually more complex than today's API, but in
practice seems to yield a much tighter API surface.

# Alternatives

Due to the known semantic problems, it is not really an option to retain the
current path implementation. As explained above, supporting UCS-2 also means
that the various byte-slice methods in the current API are untenable, so the API
also needs to change.

Probably the main alternative to the proposed API would be to *not* use
DST/slices, and instead use owned paths everywhere (probably doing some
normalization of `.` at the same time). While the resulting API would be simpler
in some respects, it would also be substantially less efficient for common operations.

# Unresolved questions

It is not clear how best to incorporate the
[WTF-8 implementation](https://github.com/SimonSapin/rust-wtf8) (or how much to
incorporate) into `libstd`.

There has been a long debate over whether paths should implement `Show` given
that they may contain non-UTF-8 data. This RFC does not take a stance on that
(the API may include something like today's `display` adapter), but a follow-up
RFC will address the question more generally.
