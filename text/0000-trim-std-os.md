- Feature Name: N/A
- Start Date: 2015-12-18
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Deprecate type aliases and structs in `std::os::$platform::raw` in favor of
trait-based accessors which return Rust types rather than the equivalent C type
aliases.

# Motivation
[motivation]: #motivation

[RFC 517][io-reform] set forth a vision for the `raw` modules in the standard
library to perform lowering operations on various Rust types to their platform
equivalents. For example the `fs::Metadata` structure can be lowered to the
underlying `sys::stat` structure. The rationale for this was to enable building
abstractions externally from the standard library by exposing all of the
underlying data that is obtained from the OS.

[io-reform]: https://github.com/rust-lang/rfcs/blob/master/text/0517-io-os-reform.md

This strategy, however, runs into a few problems:

* For some libc structures, such as `stat`, there's not actually one canonical
  definition. For example on 32-bit Linux the definition of `stat` will change
  depending on whether [LFS][lfs] is enabled (via the `-D_FILE_OFFSET_BITS`
  macro).  This means that if std is advertises these `raw` types as being "FFI
  compatible with libc", it's not actually correct in all circumstances!
* Intricately exporting raw underlying interfaces (such as [`&stat` from
  `&fs::Metadata`][std-as-stat]) makes it difficult to change the
  implementation over time.  Today the 32-bit Linux standard library [doesn't
  use LFS functions][std-no-lfs], so files over 4GB cannot be opened. Changing
  this, however, would [involve changing the `stat`
  structure][libc-stat-change] and may be difficult to do.
* Trait extensions in the `raw` module attempt to return the `libc` aliased type
  on all platforms, for example [`DirEntryExt::ino`][std-nio] returns a type of
  `ino_t`.  The `ino_t` type is billed as being FFI compatible with the libc
  `ino_t` type, but not all platforms store the `d_ino` field in `dirent` with
  the `ino_t` type. For example on Android the [definition of
  `ino_t`][android-ino_t] is `u32` but the [actual stored value is
  `u64`][android-d_ino]. This means that on Android we're actually silently
  truncating the return value!

[lfs]: http://users.suse.com/~aj/linux_lfs.html
[std-as-stat]: https://github.com/rust-lang/rust/blob/29ea4eef9fa6e36f40bc1f31eb1e56bf5941ee72/src/libstd/sys/unix/fs.rs#L81-L92
[std-no-lfs]: https://github.com/rust-lang/rust/issues/30050
[std-ino]: https://github.com/rust-lang/rust/blob/29ea4eef9fa6e36f40bc1f31eb1e56bf5941ee72/src/libstd/sys/unix/fs.rs#L192-L197
[libc-stat-change]: https://github.com/rust-lang-nursery/libc/blob/2c7e08c959e599ca221581b1670a9ecbbeac2dcb/src/unix/notbsd/linux/other/b32/mod.rs#L28-L71
[android-d_ino]: https://github.com/rust-lang-nursery/libc/blob/2c7e08c959e599ca221581b1670a9ecbbeac2dcb/src/unix/notbsd/android/mod.rs#L50
[android-ino_t]: https://github.com/rust-lang-nursery/libc/blob/2c7e08c959e599ca221581b1670a9ecbbeac2dcb/src/unix/notbsd/android/mod.rs#L11

Over time it's basically turned out that exporting the somewhat-messy details of
libc has gotten a little messy in the standard library as well. Exporting this
functionality (e.g. being able to access all of the fields), is quite useful
however! This RFC proposes tweaking the design of the extensions in
`std::os::*::raw` to allow the same level of information exposure that happens
today but also cut some of the tie from libc to std to give us more freedom to
change these implementation details and work around weird platforms.

# Detailed design
[design]: #detailed-design

First, the types and type aliases in `std::os::*::raw` will all be
deprecated. For example `stat`, `ino_t`, `dev_t`, `mode_t`, etc, will all be
deprecated (in favor of their definitions in the `libc` crate). Note that the C
integer types, `c_int` and friends, will not be deprecated.

Next, all existing extension traits will cease to return platform specific type
aliases (such as the `DirEntryExt::ino` function). Instead they will return
`u64` across the board unless it's 100% known for sure that fewer bits will
suffice. This will improve consistency across platforms as well as avoid
truncation problems such as those Android is experiencing. Furthermore this
frees std from dealing with any odd FFI compatibility issues, punting that to
the libc crate itself it the values are handed back into C.

The `std::os::*::fs::MetadataExt` will have its `as_raw_stat` method deprecated,
and it will instead grow functions to access all the associated fields of the
underlying `stat` structure. This means that there will now be a
trait-per-platform to expose all this information. Also note that all the
methods will likely return `u64` in accordance with the above modification.

With these modifications to what `std::os::*::raw` includes and how it's
defined, it should be easy to tweak existing implementations and ensure values
are transmitted in a lossless fashion. The changes, however, are both breaking
changes and don't immediately enable fixing bugs like using LFS on Linux:

* Code such as `let a: ino_t = entry.ino()` would break as the `ino()` function
  will return `u64`, but the definition of `ino_t` may not be `u64` for all
  platforms.
* The `stat` structure itself on 32-bit Linux still uses 32-bit fields (e.g. it
  doesn't mirror `stat64` in libc).

To help with these issues, more extensive modifications can be made to the
platform specific modules. All type aliases can be switched over to `u64` and
the `stat` structure could simply be redefined to `stat64` on Linux (minus
keeping the same name). This would, however, explicitly mean that
**std::os::raw is no longer FFI compatible with C**.

This breakage can be clearly indicated in the deprecation messages, however.
Additionally, this fits within std's [breaking changes policy][api-evolution] as
a local `as` cast should be all that's needed to patch code that breaks to
straddle versions of Rust.

[api-evolution]: https://github.com/rust-lang/rfcs/blob/master/text/1105-api-evolution.md

# Drawbacks
[drawbacks]: #drawbacks

As mentioned above, this RFC is strictly-speaking a breaking change. It is
expected that not much code will break, but currently there is no data
supporting this.

Returning `u64` across the board could be confusing in some circumstances as it
may wildly differ both in terms of signedness as well as size from the
underlying C type. Converting it back to the appropriate type runs the risk of
being onerous, but accessing these raw fields in theory happens quite rarely as
std should primarily be exporting cross-platform accessors for the various
fields here and there.

# Alternatives
[alternatives]: #alternatives

* The documentation of the raw modules in std could be modified to indicate that
  the types contained within are intentionally not FFI compatible, and the same
  structure could be preserved today with the types all being rewritten to what
  they would be anyway if this RFC were implemented. For example `ino_t` on
  Android would change to `u64` and `stat` on 32-bit Linux would change to
  `stat64`. In doing this, however, it's not clear why we'd keep around all the
  C namings and structure.

* Instead of breaking existing functionality, new accessors and types could be
  added to acquire the "lossless" version of a type. For example we could add a
  `ino64` function on `DirEntryExt` which returns a `u64`, and for `stat` we
  could add `as_raw_stat64`. This would, however, force `Metadata` to store two
  different `stat` structures, and the breakage in practice this will cause may
  be small enough to not warrant these great lengths.

# Unresolved questions
[unresolved]: #unresolved-questions

* Is the policy of almost always returning `u64` too strict? Should types like
  `mode_t` be allowed as `i32` explicitly? Should the sign at least attempt to
  always be preserved?
