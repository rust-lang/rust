Version 1.74.0 (2023-11-16)
==========================

<a id="1.74.0-Language"></a>

Language
--------

- [Codify that `std::mem::Discriminant<T>` does not depend on any lifetimes in T](https://github.com/rust-lang/rust/pull/104299/)
- [Replace `private_in_public` lint with `private_interfaces` and `private_bounds` per RFC 2145.](https://github.com/rust-lang/rust/pull/113126/)
  Read more in [RFC 2145](https://rust-lang.github.io/rfcs/2145-type-privacy.html).
- [Allow explicit `#[repr(Rust)]`](https://github.com/rust-lang/rust/pull/114201/)
- [closure field capturing: don't depend on alignment of packed fields](https://github.com/rust-lang/rust/pull/115315/)
- [Enable MIR-based drop-tracking for `async` blocks](https://github.com/rust-lang/rust/pull/107421/)
- [Stabilize `impl_trait_projections`](https://github.com/rust-lang/rust/pull/115659)

<a id="1.74.0-Compiler"></a>

Compiler
--------

- [stabilize combining +bundle and +whole-archive link modifiers](https://github.com/rust-lang/rust/pull/113301/)
- [Stabilize `PATH` option for `--print KIND=PATH`](https://github.com/rust-lang/rust/pull/114183/)
- [Enable ASAN/LSAN/TSAN for `*-apple-ios-macabi`](https://github.com/rust-lang/rust/pull/115644/)
- [Promote loongarch64-unknown-none* to Tier 2](https://github.com/rust-lang/rust/pull/115368/)
- [Add `i686-pc-windows-gnullvm` as a tier 3 target](https://github.com/rust-lang/rust/pull/115687/)

<a id="1.74.0-Libraries"></a>

Libraries
---------

- [Implement `From<OwnedFd/Handle>` for ChildStdin/out/err](https://github.com/rust-lang/rust/pull/98704/)
- [Implement `From<{&,&mut} [T; N]>` for `Vec<T>` where `T: Clone`](https://github.com/rust-lang/rust/pull/111278/)
- [impl Step for IP addresses](https://github.com/rust-lang/rust/pull/113748/)
- [Implement `From<[T; N]>` for `Rc<[T]>` and `Arc<[T]>`](https://github.com/rust-lang/rust/pull/114041/)
- [`impl TryFrom<char> for u16`](https://github.com/rust-lang/rust/pull/114065/)
- [Stabilize `io_error_other` feature](https://github.com/rust-lang/rust/pull/115453/)
- [Stabilize the `Saturating` type](https://github.com/rust-lang/rust/pull/115477/)
- [Stabilize const_transmute_copy](https://github.com/rust-lang/rust/pull/115520/)

<a id="1.74.0-Stabilized-APIs"></a>

Stabilized APIs
---------------

- [`core::num::Saturating`](https://doc.rust-lang.org/stable/std/num/struct.Saturating.html)
- [`impl From<io::Stdout> for std::process::Stdio`](https://doc.rust-lang.org/stable/std/process/struct.Stdio.html#impl-From%3CStdout%3E-for-Stdio)
- [`impl From<io::Stderr> for std::process::Stdio`](https://doc.rust-lang.org/stable/std/process/struct.Stdio.html#impl-From%3CStderr%3E-for-Stdio)
- [`impl From<OwnedHandle> for std::process::Child{Stdin, Stdout, Stderr}`](https://doc.rust-lang.org/stable/std/process/struct.ChildStderr.html#impl-From%3COwnedHandle%3E-for-ChildStderr)
- [`impl From<OwnedFd> for std::process::Child{Stdin, Stdout, Stderr}`](https://doc.rust-lang.org/stable/std/process/struct.ChildStderr.html#impl-From%3COwnedFd%3E-for-ChildStderr)
- [`std::ffi::OsString::from_encoded_bytes_unchecked`](https://doc.rust-lang.org/stable/std/ffi/struct.OsString.html#method.from_encoded_bytes_unchecked)
- [`std::ffi::OsString::into_encoded_bytes`](https://doc.rust-lang.org/stable/std/ffi/struct.OsString.html#method.into_encoded_bytes)
- [`std::ffi::OsStr::from_encoded_bytes_unchecked`](https://doc.rust-lang.org/stable/std/ffi/struct.OsStr.html#method.from_encoded_bytes_unchecked)
- [`std::ffi::OsStr::as_encoded_bytes`](https://doc.rust-lang.org/stable/std/ffi/struct.OsStr.html#method.as_encoded_bytes)
- [`std::io::Error::other`](https://doc.rust-lang.org/stable/std/io/struct.Error.html#method.other)
- [`impl TryFrom<char> for u16`](https://doc.rust-lang.org/stable/std/primitive.u16.html#impl-TryFrom%3Cchar%3E-for-u16)
- [`impl<T: Clone, const N: usize> From<&[T; N]> for Vec<T>`](https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#impl-From%3C%26%5BT;+N%5D%3E-for-Vec%3CT,+Global%3E)
- [`impl<T: Clone, const N: usize> From<&mut [T; N]> for Vec<T>`](https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#impl-From%3C%26mut+%5BT;+N%5D%3E-for-Vec%3CT,+Global%3E)
- [`impl<T, const N: usize> From<[T; N]> for Arc<[T]>`](https://doc.rust-lang.org/stable/std/sync/struct.Arc.html#impl-From%3C%5BT;+N%5D%3E-for-Arc%3C%5BT%5D,+Global%3E)
- [`impl<T, const N: usize> From<[T; N]> for Rc<[T]>`](https://doc.rust-lang.org/stable/std/rc/struct.Rc.html#impl-From%3C%5BT;+N%5D%3E-for-Rc%3C%5BT%5D,+Global%3E)

These APIs are now stable in const contexts:

- [`core::mem::transmute_copy`](https://doc.rust-lang.org/beta/std/mem/fn.transmute_copy.html)
- [`str::is_ascii`](https://doc.rust-lang.org/beta/std/primitive.str.html#method.is_ascii)
- [`[u8]::is_ascii`](https://doc.rust-lang.org/beta/std/primitive.slice.html#method.is_ascii)

<a id="1.74.0-Cargo"></a>

Cargo
-----

- [In `Cargo.toml`, stabilize `[lints]`](https://github.com/rust-lang/cargo/pull/12648/)
- [Stabilize credential-process and registry-auth](https://github.com/rust-lang/cargo/pull/12649/)
- [Stabilize `--keep-going` build flag](https://github.com/rust-lang/cargo/pull/12568/)
- [Add styling to `--help` output](https://github.com/rust-lang/cargo/pull/12578/)
- [For `cargo clean`, add `--dry-run` flag and summary line at the end](https://github.com/rust-lang/cargo/pull/12638)
- [For `cargo update`, make `--package` more convenient by being positional](https://github.com/rust-lang/cargo/pull/12545/)
- [For `cargo update`, clarify meaning of --aggressive as --recursive](https://github.com/rust-lang/cargo/pull/12544/)
- [Add '-n' as an alias for `--dry-run`](https://github.com/rust-lang/cargo/pull/12660/)
- [Allow version-prefixes in pkgid's (e.g. `--package` flags) to resolve ambiguities](https://github.com/rust-lang/cargo/pull/12614/)
- [In `.cargo/config.toml`, merge lists in precedence order](https://github.com/rust-lang/cargo/pull/12515/)
- [Add support for `target.'cfg(..)'.linker`](https://github.com/rust-lang/cargo/pull/12535/)

<a id="1.74.0-Rustdoc"></a>

Rustdoc
-------

- [Add warning block support in rustdoc](https://github.com/rust-lang/rust/pull/106561/)
- [Accept additional user-defined syntax classes in fenced code blocks](https://github.com/rust-lang/rust/pull/110800/)
- [rustdoc-search: add support for type parameters](https://github.com/rust-lang/rust/pull/112725/)
- [rustdoc: show inner enum and struct in type definition for concrete type](https://github.com/rust-lang/rust/pull/114855/)

<a id="1.74.0-Compatibility-Notes"></a>

Compatibility Notes
-------------------

- [Raise minimum supported Apple OS versions](https://github.com/rust-lang/rust/pull/104385/)
- [make Cell::swap panic if the Cells partially overlap](https://github.com/rust-lang/rust/pull/114795/)
- [Reject invalid crate names in `--extern`](https://github.com/rust-lang/rust/pull/116001/)
- [Don't resolve generic impls that may be shadowed by dyn built-in impls](https://github.com/rust-lang/rust/pull/114941/)
- [The new `impl From<{&,&mut} [T; N]> for Vec<T>` is known to cause some inference failures with overly-generic code.](https://github.com/rust-lang/rust/issues/117054) In those examples using the `tui` crate, the combination of `AsRef<_>` and `Into<Vec>` leaves the middle type ambiguous, and the new `impl` adds another possibility, so it now requires an explicit type annotation.

<a id="1.74.0-Internal-Changes"></a>

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.

None this cycle.

Version 1.73.0 (2023-10-05)
==========================

<a id="1.73.0-Language"></a>

Language
--------

- [Uplift `clippy::fn_null_check` lint as `useless_ptr_null_checks`.](https://github.com/rust-lang/rust/pull/111717/)
- [Make `noop_method_call` warn by default.](https://github.com/rust-lang/rust/pull/111916/)
- [Support interpolated block for `try` and `async` in macros.](https://github.com/rust-lang/rust/pull/112953/)
- [Make `unconditional_recursion` lint detect recursive drops.](https://github.com/rust-lang/rust/pull/113902/)
- [Future compatibility warning for some impls being incorrectly considered not overlapping.](https://github.com/rust-lang/rust/pull/114023/)
- [The `invalid_reference_casting` lint is now **deny-by-default** (instead of allow-by-default)](https://github.com/rust-lang/rust/pull/112431)

<a id="1.73.0-Compiler"></a>

Compiler
--------

- [Write version information in a `.comment` section like GCC/Clang.](https://github.com/rust-lang/rust/pull/97550/)
- [Add documentation on v0 symbol mangling.](https://github.com/rust-lang/rust/pull/97571/)
- [Stabilize `extern "thiscall"` and `"thiscall-unwind"` ABIs.](https://github.com/rust-lang/rust/pull/114562/)
- [Only check outlives goals on impl compared to trait.](https://github.com/rust-lang/rust/pull/109356/)
- [Infer type in irrefutable slice patterns with fixed length as array.](https://github.com/rust-lang/rust/pull/113199/)
- [Discard default auto trait impls if explicit ones exist.](https://github.com/rust-lang/rust/pull/113312/)
- Add several new tier 3 targets:
    - [`aarch64-unknown-teeos`](https://github.com/rust-lang/rust/pull/113480/)
    - [`csky-unknown-linux-gnuabiv2`](https://github.com/rust-lang/rust/pull/113658/)
    - [`riscv64-linux-android`](https://github.com/rust-lang/rust/pull/112858/)
    - [`riscv64gc-unknown-hermit`](https://github.com/rust-lang/rust/pull/114004/)
    - [`x86_64-unikraft-linux-musl`](https://github.com/rust-lang/rust/pull/113411/)
    - [`x86_64-unknown-linux-ohos`](https://github.com/rust-lang/rust/pull/113061/)
- [Add `wasm32-wasi-preview1-threads` as a tier 2 target.](https://github.com/rust-lang/rust/pull/112922/)

Refer to Rust's [platform support page][platform-support-doc]
for more information on Rust's tiered platform support.

<a id="1.73.0-Libraries"></a>

Libraries
---------

- [Add `Read`, `Write` and `Seek` impls for `Arc<File>`.](https://github.com/rust-lang/rust/pull/94748/)
- [Merge functionality of `io::Sink` into `io::Empty`.](https://github.com/rust-lang/rust/pull/98154/)
- [Implement `RefUnwindSafe` for `Backtrace`](https://github.com/rust-lang/rust/pull/100455/)
- [Make `ExitStatus` implement `Default`](https://github.com/rust-lang/rust/pull/106425/)
- [`impl SliceIndex<str> for (Bound<usize>, Bound<usize>)`](https://github.com/rust-lang/rust/pull/111081/)
- [Change default panic handler message format.](https://github.com/rust-lang/rust/pull/112849/)
- [Cleaner `assert_eq!` & `assert_ne!` panic messages.](https://github.com/rust-lang/rust/pull/111071/)
- [Correct the (deprecated) Android `stat` struct definitions.](https://github.com/rust-lang/rust/pull/113130/)

<a id="1.73.0-Stabilized-APIs"></a>

Stabilized APIs
---------------

- [Unsigned `{integer}::div_ceil`](https://doc.rust-lang.org/stable/std/primitive.u32.html#method.div_ceil)
- [Unsigned `{integer}::next_multiple_of`](https://doc.rust-lang.org/stable/std/primitive.u32.html#method.next_multiple_of)
- [Unsigned `{integer}::checked_next_multiple_of`](https://doc.rust-lang.org/stable/std/primitive.u32.html#method.checked_next_multiple_of)
- [`std::ffi::FromBytesUntilNulError`](https://doc.rust-lang.org/stable/std/ffi/struct.FromBytesUntilNulError.html)
- [`std::os::unix::fs::chown`](https://doc.rust-lang.org/stable/std/os/unix/fs/fn.chown.html)
- [`std::os::unix::fs::fchown`](https://doc.rust-lang.org/stable/std/os/unix/fs/fn.fchown.html)
- [`std::os::unix::fs::lchown`](https://doc.rust-lang.org/stable/std/os/unix/fs/fn.lchown.html)
- [`LocalKey::<Cell<T>>::get`](https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html#method.get)
- [`LocalKey::<Cell<T>>::set`](https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html#method.set)
- [`LocalKey::<Cell<T>>::take`](https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html#method.take)
- [`LocalKey::<Cell<T>>::replace`](https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html#method.replace)
- [`LocalKey::<RefCell<T>>::with_borrow`](https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html#method.with_borrow)
- [`LocalKey::<RefCell<T>>::with_borrow_mut`](https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html#method.with_borrow_mut)
- [`LocalKey::<RefCell<T>>::set`](https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html#method.set-1)
- [`LocalKey::<RefCell<T>>::take`](https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html#method.take-1)
- [`LocalKey::<RefCell<T>>::replace`](https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html#method.replace-1)

These APIs are now stable in const contexts:

- [`rc::Weak::new`](https://doc.rust-lang.org/stable/alloc/rc/struct.Weak.html#method.new)
- [`sync::Weak::new`](https://doc.rust-lang.org/stable/alloc/sync/struct.Weak.html#method.new)
- [`NonNull::as_ref`](https://doc.rust-lang.org/stable/core/ptr/struct.NonNull.html#method.as_ref)

<a id="1.73.0-Cargo"></a>

Cargo
-----

- [Bail out an error when using `cargo::` in custom build script.](https://github.com/rust-lang/cargo/pull/12332/)

<a id="1.73.0-Misc"></a>

Misc
----

<a id="1.73.0-Compatibility-Notes"></a>

Compatibility Notes
-------------------

- [Update the minimum external LLVM to 15.](https://github.com/rust-lang/rust/pull/114148/)
- [Check for non-defining uses of return position `impl Trait`.](https://github.com/rust-lang/rust/pull/112842/)

<a id="1.73.0-Internal-Changes"></a>

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.

- [Remove LLVM pointee types, supporting only opaque pointers.](https://github.com/rust-lang/rust/pull/105545/)
- [Port PGO/LTO/BOLT optimized build pipeline to Rust.](https://github.com/rust-lang/rust/pull/112235/)
- [Replace in-tree `rustc_apfloat` with the new version of the crate.](https://github.com/rust-lang/rust/pull/113843/)
- [Update to LLVM 17.](https://github.com/rust-lang/rust/pull/114048/)
- [Add `internal_features` lint for internal unstable features.](https://github.com/rust-lang/rust/pull/108955/)
- [Mention style for new syntax in tracking issue template.](https://github.com/rust-lang/rust/pull/113586/)

Version 1.72.1 (2023-09-19)
===========================

- [Adjust codegen change to improve LLVM codegen](https://github.com/rust-lang/rust/pull/115236)
- [rustdoc: Fix self ty params in objects with lifetimes](https://github.com/rust-lang/rust/pull/115276)
- [Fix regression in compile times](https://github.com/rust-lang/rust/pull/114948)
- Resolve some ICE regressions in the compiler:
  - [#115215](https://github.com/rust-lang/rust/pull/115215)
  - [#115559](https://github.com/rust-lang/rust/pull/115559)

Version 1.72.0 (2023-08-24)
==========================

<a id="1.72.0-Language"></a>

Language
--------

- [Replace const eval limit by a lint and add an exponential backoff warning](https://github.com/rust-lang/rust/pull/103877/)
- [expand: Change how `#![cfg(FALSE)]` behaves on crate root](https://github.com/rust-lang/rust/pull/110141/)
- [Stabilize inline asm for LoongArch64](https://github.com/rust-lang/rust/pull/111235/)
- [Uplift `clippy::undropped_manually_drops` lint](https://github.com/rust-lang/rust/pull/111530/)
- [Uplift `clippy::invalid_utf8_in_unchecked` lint](https://github.com/rust-lang/rust/pull/111543/) as `invalid_from_utf8_unchecked` and `invalid_from_utf8`
- [Uplift `clippy::cast_ref_to_mut` lint](https://github.com/rust-lang/rust/pull/111567/) as `invalid_reference_casting`
- [Uplift `clippy::cmp_nan` lint](https://github.com/rust-lang/rust/pull/111818/) as `invalid_nan_comparisons`
- [resolve: Remove artificial import ambiguity errors](https://github.com/rust-lang/rust/pull/112086/)
- [Don't require associated types with Self: Sized bounds in `dyn Trait` objects](https://github.com/rust-lang/rust/pull/112319/)

<a id="1.72.0-Compiler"></a>

Compiler
--------

- [Remember names of `cfg`-ed out items to mention them in diagnostics](https://github.com/rust-lang/rust/pull/109005/)
- [Support for native WASM exceptions](https://github.com/rust-lang/rust/pull/111322/)
- [Add support for NetBSD/aarch64-be (big-endian arm64).](https://github.com/rust-lang/rust/pull/111326/)
- [Write to stdout if `-` is given as output file](https://github.com/rust-lang/rust/pull/111626/)
- [Force all native libraries to be statically linked when linking a static binary](https://github.com/rust-lang/rust/pull/111698/)
- [Add Tier 3 support for `loongarch64-unknown-none*`](https://github.com/rust-lang/rust/pull/112310/)
- [Prevent `.eh_frame` from being emitted for `-C panic=abort`](https://github.com/rust-lang/rust/pull/112403/)
- [Support 128-bit enum variant in debuginfo codegen](https://github.com/rust-lang/rust/pull/112474/)
- [compiler: update solaris/illumos to enable tsan support.](https://github.com/rust-lang/rust/pull/112039/)

Refer to Rust's [platform support page][platform-support-doc]
for more information on Rust's tiered platform support.

<a id="1.72.0-Libraries"></a>

Libraries
---------

- [Document memory orderings of `thread::{park, unpark}`](https://github.com/rust-lang/rust/pull/99587/)
- [io: soften ‘at most one write attempt’ requirement in io::Write::write](https://github.com/rust-lang/rust/pull/107200/)
- [Specify behavior of HashSet::insert](https://github.com/rust-lang/rust/pull/107619/)
- [Relax implicit `T: Sized` bounds on `BufReader<T>`, `BufWriter<T>` and `LineWriter<T>`](https://github.com/rust-lang/rust/pull/111074/)
- [Update runtime guarantee for `select_nth_unstable`](https://github.com/rust-lang/rust/pull/111974/)
- [Return `Ok` on kill if process has already exited](https://github.com/rust-lang/rust/pull/112594/)
- [Implement PartialOrd for `Vec`s over different allocators](https://github.com/rust-lang/rust/pull/112632/)
- [Use 128 bits for TypeId hash](https://github.com/rust-lang/rust/pull/109953/)
- [Don't drain-on-drop in DrainFilter impls of various collections.](https://github.com/rust-lang/rust/pull/104455/)
- [Make `{Arc,Rc,Weak}::ptr_eq` ignore pointer metadata](https://github.com/rust-lang/rust/pull/106450/)

<a id="1.72.0-Rustdoc"></a>

Rustdoc
-------

- [Allow whitespace as path separator like double colon](https://github.com/rust-lang/rust/pull/108537/)
- [Add search result item types after their name](https://github.com/rust-lang/rust/pull/110688/)
- [Search for slices and arrays by type with `[]`](https://github.com/rust-lang/rust/pull/111958/)
- [Clean up type unification and "unboxing"](https://github.com/rust-lang/rust/pull/112233/)

<a id="1.72.0-Stabilized-APIs"></a>

Stabilized APIs
---------------

- [`impl<T: Send> Sync for mpsc::Sender<T>`](https://doc.rust-lang.org/stable/std/sync/mpsc/struct.Sender.html#impl-Sync-for-Sender%3CT%3E)
- [`impl TryFrom<&OsStr> for &str`](https://doc.rust-lang.org/stable/std/primitive.str.html#impl-TryFrom%3C%26'a+OsStr%3E-for-%26'a+str)
- [`String::leak`](https://doc.rust-lang.org/stable/alloc/string/struct.String.html#method.leak)

These APIs are now stable in const contexts:

- [`CStr::from_bytes_with_nul`](https://doc.rust-lang.org/stable/std/ffi/struct.CStr.html#method.from_bytes_with_nul)
- [`CStr::to_bytes`](https://doc.rust-lang.org/stable/std/ffi/struct.CStr.html#method.to_bytes)
- [`CStr::to_bytes_with_nul`](https://doc.rust-lang.org/stable/std/ffi/struct.CStr.html#method.to_bytes_with_nul)
- [`CStr::to_str`](https://doc.rust-lang.org/stable/std/ffi/struct.CStr.html#method.to_str)

<a id="1.72.0-Cargo"></a>

Cargo
-----

- Enable `-Zdoctest-in-workspace` by default. When running each documentation
  test, the working directory is set to the root directory of the package the
  test belongs to.
  [docs](https://doc.rust-lang.org/nightly/cargo/commands/cargo-test.html#working-directory-of-tests)
  [#12221](https://github.com/rust-lang/cargo/pull/12221)
  [#12288](https://github.com/rust-lang/cargo/pull/12288)
- Add support of the "default" keyword to reset previously set `build.jobs`
  parallelism back to the default.
  [#12222](https://github.com/rust-lang/cargo/pull/12222)

<a id="1.72.0-Compatibility-Notes"></a>

Compatibility Notes
-------------------

- [Alter `Display` for `Ipv6Addr` for IPv4-compatible addresses](https://github.com/rust-lang/rust/pull/112606/)
- Cargo changed feature name validation check to a hard error. The warning was
  added in Rust 1.49. These extended characters aren't allowed on crates.io, so
  this should only impact users of other registries, or people who don't publish
  to a registry.
  [#12291](https://github.com/rust-lang/cargo/pull/12291)
- [Demoted `mips*-unknown-linux-gnu*` targets from host tier 2 to target tier 3 support.](https://github.com/rust-lang/rust/pull/113274)

Version 1.71.1 (2023-08-03)
===========================

- [Fix CVE-2023-38497: Cargo did not respect the umask when extracting dependencies](https://github.com/rust-lang/cargo/security/advisories/GHSA-j3xp-wfr4-hx87)
- [Fix bash completion for users of Rustup](https://github.com/rust-lang/rust/pull/113579)
- [Do not show `suspicious_double_ref_op` lint when calling `borrow()`](https://github.com/rust-lang/rust/pull/112517)
- [Fix ICE: substitute types before checking inlining compatibility](https://github.com/rust-lang/rust/pull/113802)
- [Fix ICE: don't use `can_eq` in `derive(..)` suggestion for missing method](https://github.com/rust-lang/rust/pull/111516)
- [Fix building Rust 1.71.0 from the source tarball](https://github.com/rust-lang/rust/issues/113678)

Version 1.71.0 (2023-07-13)
==========================

<a id="1.71.0-Language"></a>

Language
--------

- [Stabilize `raw-dylib`, `link_ordinal`, `import_name_type` and `-Cdlltool`.](https://github.com/rust-lang/rust/pull/109677/)
- [Uplift `clippy::{drop,forget}_{ref,copy}` lints.](https://github.com/rust-lang/rust/pull/109732/)
- [Type inference is more conservative around constrained vars.](https://github.com/rust-lang/rust/pull/110100/)
- [Use fulfillment to check `Drop` impl compatibility](https://github.com/rust-lang/rust/pull/110577/)

<a id="1.71.0-Compiler"></a>

Compiler
--------

- [Evaluate place expression in `PlaceMention`](https://github.com/rust-lang/rust/pull/104844/),
  making `let _ =` patterns more consistent with respect to the borrow checker.
- [Add `--print deployment-target` flag for Apple targets.](https://github.com/rust-lang/rust/pull/105354/)
- [Stabilize `extern "C-unwind"` and friends.](https://github.com/rust-lang/rust/pull/106075/)
  The existing `extern "C"` etc. may change behavior for cross-language unwinding in a future release.
- [Update the version of musl used on `*-linux-musl` targets to 1.2.3](https://github.com/rust-lang/rust/pull/107129/),
  enabling [time64](https://musl.libc.org/time64.html) on 32-bit systems.
- [Stabilize `debugger_visualizer`](https://github.com/rust-lang/rust/pull/108668/)
  for embedding metadata like Microsoft's Natvis.
- [Enable flatten-format-args by default.](https://github.com/rust-lang/rust/pull/109999/)
- [Make `Self` respect tuple constructor privacy.](https://github.com/rust-lang/rust/pull/111245/)
- [Improve niche placement by trying two strategies and picking the better result.](https://github.com/rust-lang/rust/pull/108106/)
- [Use `apple-m1` as the target CPU for `aarch64-apple-darwin`.](https://github.com/rust-lang/rust/pull/109899/)
- [Add Tier 3 support for the `x86_64h-apple-darwin` target.](https://github.com/rust-lang/rust/pull/108795/)
- [Promote `loongarch64-unknown-linux-gnu` to Tier 2 with host tools.](https://github.com/rust-lang/rust/pull/110936/)

Refer to Rust's [platform support page][platform-support-doc]
for more information on Rust's tiered platform support.

<a id="1.71.0-Libraries"></a>

Libraries
---------
- [Rework handling of recursive panics.](https://github.com/rust-lang/rust/pull/110975/)
  Additional panics are allowed while unwinding, as long as they are caught before escaping
  a `Drop` implementation, but panicking within a panic hook is now an immediate abort.
- [Loosen `From<&[T]> for Box<[T]>` bound to `T: Clone`.](https://github.com/rust-lang/rust/pull/103406/)
- [Remove unnecessary `T: Send` bound](https://github.com/rust-lang/rust/pull/111134/)
  in `Error for mpsc::SendError<T>` and `TrySendError<T>`.
- [Fix docs for `alloc::realloc`](https://github.com/rust-lang/rust/pull/108630/)
  to match `Layout` requirements that the size must not exceed `isize::MAX`.
- [Document `const {}` syntax for `std::thread_local`.](https://github.com/rust-lang/rust/pull/110620/)
  This syntax was stabilized in Rust 1.59, but not previously mentioned in release notes.

<a id="1.71.0-Stabilized-APIs"></a>

Stabilized APIs
---------------

- [`CStr::is_empty`](https://doc.rust-lang.org/stable/std/ffi/struct.CStr.html#method.is_empty)
- [`BuildHasher::hash_one`](https://doc.rust-lang.org/stable/std/hash/trait.BuildHasher.html#method.hash_one)
- [`NonZeroI*::is_positive`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroI32.html#method.is_positive)
- [`NonZeroI*::is_negative`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroI32.html#method.is_negative)
- [`NonZeroI*::checked_neg`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroI32.html#method.checked_neg)
- [`NonZeroI*::overflowing_neg`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroI32.html#method.overflowing_neg)
- [`NonZeroI*::saturating_neg`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroI32.html#method.saturating_neg)
- [`NonZeroI*::wrapping_neg`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroI32.html#method.wrapping_neg)
- [`Neg for NonZeroI*`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroI32.html#impl-Neg-for-NonZeroI32)
- [`Neg for &NonZeroI*`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroI32.html#impl-Neg-for-%26NonZeroI32)
- [`From<[T; N]> for (T...)`](https://doc.rust-lang.org/stable/std/primitive.array.html#impl-From%3C%5BT;+1%5D%3E-for-(T,))
  (array to N-tuple for N in 1..=12)
- [`From<(T...)> for [T; N]`](https://doc.rust-lang.org/stable/std/primitive.array.html#impl-From%3C(T,)%3E-for-%5BT;+1%5D)
  (N-tuple to array for N in 1..=12)
- [`windows::io::AsHandle for Box<T>`](https://doc.rust-lang.org/stable/std/os/windows/io/trait.AsHandle.html#impl-AsHandle-for-Box%3CT%3E)
- [`windows::io::AsHandle for Rc<T>`](https://doc.rust-lang.org/stable/std/os/windows/io/trait.AsHandle.html#impl-AsHandle-for-Rc%3CT%3E)
- [`windows::io::AsHandle for Arc<T>`](https://doc.rust-lang.org/stable/std/os/windows/io/trait.AsHandle.html#impl-AsHandle-for-Arc%3CT%3E)
- [`windows::io::AsSocket for Box<T>`](https://doc.rust-lang.org/stable/std/os/windows/io/trait.AsSocket.html#impl-AsSocket-for-Box%3CT%3E)
- [`windows::io::AsSocket for Rc<T>`](https://doc.rust-lang.org/stable/std/os/windows/io/trait.AsSocket.html#impl-AsSocket-for-Rc%3CT%3E)
- [`windows::io::AsSocket for Arc<T>`](https://doc.rust-lang.org/stable/std/os/windows/io/trait.AsSocket.html#impl-AsSocket-for-Arc%3CT%3E)

These APIs are now stable in const contexts:

- [`<*const T>::read`](https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.read)
- [`<*const T>::read_unaligned`](https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.read_unaligned)
- [`<*mut T>::read`](https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.read-1)
- [`<*mut T>::read_unaligned`](https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.read_unaligned-1)
- [`ptr::read`](https://doc.rust-lang.org/stable/std/ptr/fn.read.html)
- [`ptr::read_unaligned`](https://doc.rust-lang.org/stable/std/ptr/fn.read_unaligned.html)
- [`<[T]>::split_at`](https://doc.rust-lang.org/stable/std/primitive.slice.html#method.split_at)

<a id="1.71.0-Cargo"></a>

Cargo
-----
- [Allow named debuginfo options in `Cargo.toml`.](https://github.com/rust-lang/cargo/pull/11958/)
- [Add `workspace_default_members` to the output of `cargo metadata`.](https://github.com/rust-lang/cargo/pull/11978/)
- [Automatically inherit workspace fields when running `cargo new`/`cargo init`.](https://github.com/rust-lang/cargo/pull/12069/)

<a id="1.71.0-Rustdoc"></a>

Rustdoc
-------

- [Add a new `rustdoc::unescaped_backticks` lint for broken inline code.](https://github.com/rust-lang/rust/pull/105848/)
- [Support strikethrough with single tildes.](https://github.com/rust-lang/rust/pull/111152/) (`~~old~~` vs. `~new~`)

<a id="1.71.0-Misc"></a>

Misc
----

<a id="1.71.0-Compatibility-Notes"></a>

Compatibility Notes
-------------------

- [Remove structural match from `TypeId`.](https://github.com/rust-lang/rust/pull/103291/)
  Code that uses a constant `TypeId` in a pattern will potentially be broken.
  Known cases have already been fixed -- in particular, users of the `log`
  crate's `kv_unstable` feature should update to `log v0.4.18` or later.
- [Add a `sysroot` crate to represent the standard library crates.](https://github.com/rust-lang/rust/pull/108865/)
  This does not affect stable users, but may require adjustment in tools that build their own standard library.
- [Cargo optimizes its usage under `rustup`.](https://github.com/rust-lang/cargo/pull/11917/) When
  Cargo detects it will run `rustc` pointing to a rustup proxy, it'll try bypassing the proxy and
  use the underlying binary directly. There are assumptions around the interaction with rustup and
  `RUSTUP_TOOLCHAIN`. However, it's not expected to affect normal users.
- [When querying a package, Cargo tries only the original name, all hyphens, and all underscores to
  handle misspellings.](https://github.com/rust-lang/cargo/pull/12083/) Previously, Cargo tried each
  combination of hyphens and underscores, causing excessive requests to crates.io.
- Cargo now [disallows `RUSTUP_HOME`](https://github.com/rust-lang/cargo/pull/12101/) and
  [`RUSTUP_TOOLCHAIN`](https://github.com/rust-lang/cargo/pull/12107/) in the `[env]` configuration
  table. This is considered to be not a use case Cargo would like to support, since it will likely
  cause problems or lead to confusion.

<a id="1.71.0-Internal-Changes"></a>

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.


Version 1.70.0 (2023-06-01)
==========================

<a id="1.70.0-Language"></a>

Language
--------
- [Relax ordering rules for `asm!` operands](https://github.com/rust-lang/rust/pull/105798/)
- [Properly allow macro expanded `format_args` invocations to uses captures](https://github.com/rust-lang/rust/pull/106505/)
- [Lint ambiguous glob re-exports](https://github.com/rust-lang/rust/pull/107880/)
- [Perform const and unsafe checking for expressions in `let _ = expr` position.](https://github.com/rust-lang/rust/pull/102256/)

<a id="1.70.0-Compiler"></a>

Compiler
--------
- [Extend -Cdebuginfo with new options and named aliases](https://github.com/rust-lang/rust/pull/109808/)
  This provides a smaller version of debuginfo for cases that only need line number information
  (`-Cdebuginfo=line-tables-only`), which may eventually become the default for `-Cdebuginfo=1`.
- [Make `unused_allocation` lint against `Box::new` too](https://github.com/rust-lang/rust/pull/104363/)
- [Detect uninhabited types early in const eval](https://github.com/rust-lang/rust/pull/109435/)
- [Switch to LLD as default linker for {arm,thumb}v4t-none-eabi](https://github.com/rust-lang/rust/pull/109721/)
- [Add tier 3 target `loongarch64-unknown-linux-gnu`](https://github.com/rust-lang/rust/pull/96971)
- [Add tier 3 target for `i586-pc-nto-qnx700` (QNX Neutrino RTOS, version 7.0)](https://github.com/rust-lang/rust/pull/109173/), 
- [Insert alignment checks for pointer dereferences as debug assertions](https://github.com/rust-lang/rust/pull/98112)
  This catches undefined behavior at runtime, and may cause existing code to fail.

Refer to Rust's [platform support page][platform-support-doc]
for more information on Rust's tiered platform support.

<a id="1.70.0-Libraries"></a>

Libraries
---------
- [Document NonZeroXxx layout guarantees](https://github.com/rust-lang/rust/pull/94786/)
- [Windows: make `Command` prefer non-verbatim paths](https://github.com/rust-lang/rust/pull/96391/)
- [Implement Default for some alloc/core iterators](https://github.com/rust-lang/rust/pull/99929/)
- [Fix handling of trailing bare CR in str::lines](https://github.com/rust-lang/rust/pull/100311/)
- [allow negative numeric literals in `concat!`](https://github.com/rust-lang/rust/pull/106844/)
- [Add documentation about the memory layout of `Cell`](https://github.com/rust-lang/rust/pull/106921/)
- [Use `partial_cmp` to implement tuple `lt`/`le`/`ge`/`gt`](https://github.com/rust-lang/rust/pull/108157/)
- [Stabilize `atomic_as_ptr`](https://github.com/rust-lang/rust/pull/108419/)
- [Stabilize `nonnull_slice_from_raw_parts`](https://github.com/rust-lang/rust/pull/97506/)
- [Partial stabilization of `once_cell`](https://github.com/rust-lang/rust/pull/105587/)
- [Stabilize `nonzero_min_max`](https://github.com/rust-lang/rust/pull/106633/)
- [Flatten/inline format_args!() and (string and int) literal arguments into format_args!()](https://github.com/rust-lang/rust/pull/106824/)
- [Stabilize movbe target feature](https://github.com/rust-lang/rust/pull/107711/)
- [don't splice from files into pipes in io::copy](https://github.com/rust-lang/rust/pull/108283/)
- [Add a builtin unstable `FnPtr` trait that is implemented for all function pointers](https://github.com/rust-lang/rust/pull/108080/)
  This extends `Debug`, `Pointer`, `Hash`, `PartialEq`, `Eq`, `PartialOrd`, and `Ord`
  implementations for function pointers with all ABIs.

<a id="1.70.0-Stabilized-APIs"></a>

Stabilized APIs
---------------

- [`NonZero*::MIN/MAX`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroI8.html#associatedconstant.MIN)
- [`BinaryHeap::retain`](https://doc.rust-lang.org/stable/std/collections/struct.BinaryHeap.html#method.retain)
- [`Default for std::collections::binary_heap::IntoIter`](https://doc.rust-lang.org/stable/std/collections/binary_heap/struct.IntoIter.html)
- [`Default for std::collections::btree_map::{IntoIter, Iter, IterMut}`](https://doc.rust-lang.org/stable/std/collections/btree_map/struct.IntoIter.html)
- [`Default for std::collections::btree_map::{IntoKeys, Keys}`](https://doc.rust-lang.org/stable/std/collections/btree_map/struct.IntoKeys.html)
- [`Default for std::collections::btree_map::{IntoValues, Values}`](https://doc.rust-lang.org/stable/std/collections/btree_map/struct.IntoValues.html)
- [`Default for std::collections::btree_map::Range`](https://doc.rust-lang.org/stable/std/collections/btree_map/struct.Range.html)
- [`Default for std::collections::btree_set::{IntoIter, Iter}`](https://doc.rust-lang.org/stable/std/collections/btree_set/struct.IntoIter.html)
- [`Default for std::collections::btree_set::Range`](https://doc.rust-lang.org/stable/std/collections/btree_set/struct.Range.html)
- [`Default for std::collections::linked_list::{IntoIter, Iter, IterMut}`](https://doc.rust-lang.org/stable/alloc/collections/linked_list/struct.IntoIter.html)
- [`Default for std::vec::IntoIter`](https://doc.rust-lang.org/stable/alloc/vec/struct.IntoIter.html#impl-Default-for-IntoIter%3CT,+A%3E)
- [`Default for std::iter::Chain`](https://doc.rust-lang.org/stable/std/iter/struct.Chain.html)
- [`Default for std::iter::Cloned`](https://doc.rust-lang.org/stable/std/iter/struct.Cloned.html)
- [`Default for std::iter::Copied`](https://doc.rust-lang.org/stable/std/iter/struct.Copied.html)
- [`Default for std::iter::Enumerate`](https://doc.rust-lang.org/stable/std/iter/struct.Enumerate.html)
- [`Default for std::iter::Flatten`](https://doc.rust-lang.org/stable/std/iter/struct.Flatten.html)
- [`Default for std::iter::Fuse`](https://doc.rust-lang.org/stable/std/iter/struct.Fuse.html)
- [`Default for std::iter::Rev`](https://doc.rust-lang.org/stable/std/iter/struct.Rev.html)
- [`Default for std::slice::Iter`](https://doc.rust-lang.org/stable/std/slice/struct.Iter.html)
- [`Default for std::slice::IterMut`](https://doc.rust-lang.org/stable/std/slice/struct.IterMut.html)
- [`Rc::into_inner`](https://doc.rust-lang.org/stable/alloc/rc/struct.Rc.html#method.into_inner)
- [`Arc::into_inner`](https://doc.rust-lang.org/stable/alloc/sync/struct.Arc.html#method.into_inner)
- [`std::cell::OnceCell`](https://doc.rust-lang.org/stable/std/cell/struct.OnceCell.html)
- [`Option::is_some_and`](https://doc.rust-lang.org/stable/std/option/enum.Option.html#method.is_some_and)
- [`NonNull::slice_from_raw_parts`](https://doc.rust-lang.org/stable/std/ptr/struct.NonNull.html#method.slice_from_raw_parts)
- [`Result::is_ok_and`](https://doc.rust-lang.org/stable/std/result/enum.Result.html#method.is_ok_and)
- [`Result::is_err_and`](https://doc.rust-lang.org/stable/std/result/enum.Result.html#method.is_err_and)
- [`std::sync::atomic::Atomic*::as_ptr`](https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicU8.html#method.as_ptr)
- [`std::io::IsTerminal`](https://doc.rust-lang.org/stable/std/io/trait.IsTerminal.html)
- [`std::os::linux::net::SocketAddrExt`](https://doc.rust-lang.org/stable/std/os/linux/net/trait.SocketAddrExt.html)
- [`std::os::unix::net::UnixDatagram::bind_addr`](https://doc.rust-lang.org/stable/std/os/unix/net/struct.UnixDatagram.html#method.bind_addr)
- [`std::os::unix::net::UnixDatagram::connect_addr`](https://doc.rust-lang.org/stable/std/os/unix/net/struct.UnixDatagram.html#method.connect_addr)
- [`std::os::unix::net::UnixDatagram::send_to_addr`](https://doc.rust-lang.org/stable/std/os/unix/net/struct.UnixDatagram.html#method.send_to_addr)
- [`std::os::unix::net::UnixListener::bind_addr`](https://doc.rust-lang.org/stable/std/os/unix/net/struct.UnixListener.html#method.bind_addr)
- [`std::path::Path::as_mut_os_str`](https://doc.rust-lang.org/stable/std/path/struct.Path.html#method.as_mut_os_str)
- [`std::sync::OnceLock`](https://doc.rust-lang.org/stable/std/sync/struct.OnceLock.html)

<a id="1.70.0-Cargo"></a>

Cargo
-----

- [Add `CARGO_PKG_README`](https://github.com/rust-lang/cargo/pull/11645/)
- [Make `sparse` the default protocol for crates.io](https://github.com/rust-lang/cargo/pull/11791/)
- [Accurately show status when downgrading dependencies](https://github.com/rust-lang/cargo/pull/11839/)
- [Use registry.default for login/logout](https://github.com/rust-lang/cargo/pull/11949/)
- [Stabilize `cargo logout`](https://github.com/rust-lang/cargo/pull/11950/)

<a id="1.70.0-Misc"></a>

Misc
----

- [Stabilize rustdoc `--test-run-directory`](https://github.com/rust-lang/rust/pull/103682/)

<a id="1.70.0-Compatibility-Notes"></a>

Compatibility Notes
-------------------

- [Prevent stable `libtest` from supporting `-Zunstable-options`](https://github.com/rust-lang/rust/pull/109044/)
- [Perform const and unsafe checking for expressions in `let _ = expr` position.](https://github.com/rust-lang/rust/pull/102256/)
- [WebAssembly targets enable `sign-ext` and `mutable-globals` features in codegen](https://github.com/rust-lang/rust/issues/109807)
  This may cause incompatibility with older execution environments.
- [Insert alignment checks for pointer dereferences as debug assertions](https://github.com/rust-lang/rust/pull/98112)
  This catches undefined behavior at runtime, and may cause existing code to fail.

<a id="1.70.0-Internal-Changes"></a>

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.

- [Upgrade to LLVM 16](https://github.com/rust-lang/rust/pull/109474/)
- [Use SipHash-1-3 instead of SipHash-2-4 for StableHasher](https://github.com/rust-lang/rust/pull/107925/)

Version 1.69.0 (2023-04-20)
==========================

<a id="1.69.0-Language"></a>

Language
--------

- [Deriving built-in traits on packed structs works with `Copy` fields.](https://github.com/rust-lang/rust/pull/104429/)
- [Stabilize the `cmpxchg16b` target feature on x86 and x86_64.](https://github.com/rust-lang/rust/pull/106774/)
- [Improve analysis of trait bounds for associated types.](https://github.com/rust-lang/rust/pull/103695/)
- [Allow associated types to be used as union fields.](https://github.com/rust-lang/rust/pull/106938/)
- [Allow `Self: Autotrait` bounds on dyn-safe trait methods.](https://github.com/rust-lang/rust/pull/107082/)
- [Treat `str` as containing `[u8]` for auto trait purposes.](https://github.com/rust-lang/rust/pull/107941/)

<a id="1.69.0-Compiler"></a>

Compiler
--------

- [Upgrade `*-pc-windows-gnu` on CI to mingw-w64 v10 and GCC 12.2.](https://github.com/rust-lang/rust/pull/100178/)
- [Rework min_choice algorithm of member constraints.](https://github.com/rust-lang/rust/pull/105300/)
- [Support `true` and `false` as boolean flags in compiler arguments.](https://github.com/rust-lang/rust/pull/107043/)
- [Default `repr(C)` enums to `c_int` size.](https://github.com/rust-lang/rust/pull/107592/)

<a id="1.69.0-Libraries"></a>

Libraries
---------

- [Implement the unstable `DispatchFromDyn` for cell types, allowing downstream experimentation with custom method receivers.](https://github.com/rust-lang/rust/pull/97373/)
- [Document that `fmt::Arguments::as_str()` may return `Some(_)` in more cases after optimization, subject to change.](https://github.com/rust-lang/rust/pull/106823/)
- [Implement `AsFd` and `AsRawFd` for `Rc`.](https://github.com/rust-lang/rust/pull/107317/)

<a id="1.69.0-Stabilized-APIs"></a>

Stabilized APIs
---------------

- [`CStr::from_bytes_until_nul`](https://doc.rust-lang.org/stable/core/ffi/struct.CStr.html#method.from_bytes_until_nul)
- [`core::ffi::FromBytesUntilNulError`](https://doc.rust-lang.org/stable/core/ffi/struct.FromBytesUntilNulError.html)

These APIs are now stable in const contexts:

- [`SocketAddr::new`](https://doc.rust-lang.org/stable/std/net/enum.SocketAddr.html#method.new)
- [`SocketAddr::ip`](https://doc.rust-lang.org/stable/std/net/enum.SocketAddr.html#method.ip)
- [`SocketAddr::port`](https://doc.rust-lang.org/stable/std/net/enum.SocketAddr.html#method.port)
- [`SocketAddr::is_ipv4`](https://doc.rust-lang.org/stable/std/net/enum.SocketAddr.html#method.is_ipv4)
- [`SocketAddr::is_ipv6`](https://doc.rust-lang.org/stable/std/net/enum.SocketAddr.html#method.is_ipv6)
- [`SocketAddrV4::new`](https://doc.rust-lang.org/stable/std/net/struct.SocketAddrV4.html#method.new)
- [`SocketAddrV4::ip`](https://doc.rust-lang.org/stable/std/net/struct.SocketAddrV4.html#method.ip)
- [`SocketAddrV4::port`](https://doc.rust-lang.org/stable/std/net/struct.SocketAddrV4.html#method.port)
- [`SocketAddrV6::new`](https://doc.rust-lang.org/stable/std/net/struct.SocketAddrV6.html#method.new)
- [`SocketAddrV6::ip`](https://doc.rust-lang.org/stable/std/net/struct.SocketAddrV6.html#method.ip)
- [`SocketAddrV6::port`](https://doc.rust-lang.org/stable/std/net/struct.SocketAddrV6.html#method.port)
- [`SocketAddrV6::flowinfo`](https://doc.rust-lang.org/stable/std/net/struct.SocketAddrV6.html#method.flowinfo)
- [`SocketAddrV6::scope_id`](https://doc.rust-lang.org/stable/std/net/struct.SocketAddrV6.html#method.scope_id)

<a id="1.69.0-Cargo"></a>

Cargo
-----

- [Cargo now suggests `cargo fix` or `cargo clippy --fix` when compilation warnings are auto-fixable.](https://github.com/rust-lang/cargo/pull/11558/)
- [Cargo now suggests `cargo add` if you try to install a library crate.](https://github.com/rust-lang/cargo/pull/11410/)
- [Cargo now sets the `CARGO_BIN_NAME` environment variable also for binary examples.](https://github.com/rust-lang/cargo/pull/11705/)

<a id="1.69.0-Rustdoc"></a>

Rustdoc
-----

- [Vertically compact trait bound formatting.](https://github.com/rust-lang/rust/pull/102842/)
- [Only include stable lints in `rustdoc::all` group.](https://github.com/rust-lang/rust/pull/106316/)
- [Compute maximum Levenshtein distance based on the query.](https://github.com/rust-lang/rust/pull/107141/)
- [Remove inconsistently-present sidebar tooltips.](https://github.com/rust-lang/rust/pull/107490/)
- [Search by macro when query ends with `!`.](https://github.com/rust-lang/rust/pull/108143/)

<a id="1.69.0-Compatibility-Notes"></a>

Compatibility Notes
-------------------

- [The `rust-analysis` component from `rustup` now only contains a warning placeholder.](https://github.com/rust-lang/rust/pull/101841/) This was primarily intended for RLS, and the corresponding `-Zsave-analysis` flag has been removed from the compiler as well.
- [Unaligned references to packed fields are now a hard error.](https://github.com/rust-lang/rust/pull/102513/) This has been a warning since 1.53, and denied by default with a future-compatibility warning since 1.62.
- [Update the minimum external LLVM to 14.](https://github.com/rust-lang/rust/pull/107573/)
- [Cargo now emits errors on invalid characters in a registry token.](https://github.com/rust-lang/cargo/pull/11600/)
- [When `default-features` is set to false of a workspace dependency, and an inherited dependency of a member has `default-features = true`, Cargo will enable default features of that dependency.](https://github.com/rust-lang/cargo/pull/11409/)
- [Cargo denies `CARGO_HOME` in the `[env]` configuration table. Cargo itself doesn't pick up this value, but recursive calls to cargo would, which was not intended.](https://github.com/rust-lang/cargo/pull/11644/)
- [Debuginfo for build dependencies is now off if not explicitly set. This is expected to improve the overall build time.](https://github.com/rust-lang/cargo/pull/11252/)
- [The Rust distribution no longer always includes rustdoc](https://github.com/rust-lang/rust/pull/106886)
  If `tools = [...]` is set in config.toml, we will respect a missing rustdoc in that list. By
  default rustdoc remains included. To retain the prior behavior explicitly add `"rustdoc"` to the
  list.
  
<a id="1.69.0-Internal-Changes"></a>

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.

- [Move `format_args!()` into AST (and expand it during AST lowering)](https://github.com/rust-lang/rust/pull/106745/)

Version 1.68.2 (2023-03-28)
===========================

- [Update the GitHub RSA host key bundled within Cargo](https://github.com/rust-lang/cargo/pull/11883).
  The key was [rotated by GitHub](https://github.blog/2023-03-23-we-updated-our-rsa-ssh-host-key/)
  on 2023-03-24 after the old one leaked.
- [Mark the old GitHub RSA host key as revoked](https://github.com/rust-lang/cargo/pull/11889).
  This will prevent Cargo from accepting the leaked key even when trusted by
  the system.
- [Add support for `@revoked` and a better error message for `@cert-authority` in Cargo's SSH host key verification](https://github.com/rust-lang/cargo/pull/11635)

Version 1.68.1 (2023-03-23)
===========================

- [Fix miscompilation in produced Windows MSVC artifacts](https://github.com/rust-lang/rust/pull/109094)
  This was introduced by enabling ThinLTO for the distributed rustc which led
  to miscompilations in the resulting binary. Currently this is believed to be
  limited to the -Zdylib-lto flag used for rustc compilation, rather than a
  general bug in ThinLTO, so only rustc artifacts should be affected.
- [Fix --enable-local-rust builds](https://github.com/rust-lang/rust/pull/109111/)
- [Treat `$prefix-clang` as `clang` in linker detection code](https://github.com/rust-lang/rust/pull/109156)
- [Fix panic in compiler code](https://github.com/rust-lang/rust/pull/108162)

Version 1.68.0 (2023-03-09)
==========================

<a id="1.68.0-Language"></a>

Language
--------

- [Stabilize default_alloc_error_handler](https://github.com/rust-lang/rust/pull/102318/)
  This allows usage of `alloc` on stable without requiring the 
  definition of a handler for allocation failure. Defining custom handlers is still unstable.
- [Stabilize `efiapi` calling convention.](https://github.com/rust-lang/rust/pull/105795/)
- [Remove implicit promotion for types with drop glue](https://github.com/rust-lang/rust/pull/105085/)

<a id="1.68.0-Compiler"></a>

Compiler
--------

- [Change `bindings_with_variant_name` to deny-by-default](https://github.com/rust-lang/rust/pull/104154/)
- [Allow .. to be parsed as let initializer](https://github.com/rust-lang/rust/pull/105701/)
- [Add `armv7-sony-vita-newlibeabihf` as a tier 3 target](https://github.com/rust-lang/rust/pull/105712/)
- [Always check alignment during compile-time const evaluation](https://github.com/rust-lang/rust/pull/104616/)
- [Disable "split dwarf inlining" by default.](https://github.com/rust-lang/rust/pull/106709/)
- [Add vendor to Fuchsia's target triple](https://github.com/rust-lang/rust/pull/106429/)
- [Enable sanitizers for s390x-linux](https://github.com/rust-lang/rust/pull/107127/)

<a id="1.68.0-Libraries"></a>

Libraries
---------

- [Loosen the bound on the Debug implementation of Weak.](https://github.com/rust-lang/rust/pull/90291/)
- [Make `std::task::Context` !Send and !Sync](https://github.com/rust-lang/rust/pull/95985/)
- [PhantomData layout guarantees](https://github.com/rust-lang/rust/pull/104081/)
- [Don't derive Debug for `OnceWith` & `RepeatWith`](https://github.com/rust-lang/rust/pull/104163/)
- [Implement DerefMut for PathBuf](https://github.com/rust-lang/rust/pull/105018/)
- [Add O(1) `Vec -> VecDeque` conversion guarantee](https://github.com/rust-lang/rust/pull/105128/)
- [Leak amplification for peek_mut() to ensure BinaryHeap's invariant is always met](https://github.com/rust-lang/rust/pull/105851/)

<a id="1.68.0-Stabilized-APIs"></a>

Stabilized APIs
---------------

- [`{core,std}::pin::pin!`](https://doc.rust-lang.org/stable/std/pin/macro.pin.html)
- [`impl From<bool> for {f32,f64}`](https://doc.rust-lang.org/stable/std/primitive.f32.html#impl-From%3Cbool%3E-for-f32)
- [`std::path::MAIN_SEPARATOR_STR`](https://doc.rust-lang.org/stable/std/path/constant.MAIN_SEPARATOR_STR.html)
- [`impl DerefMut for PathBuf`](https://doc.rust-lang.org/stable/std/path/struct.PathBuf.html#impl-DerefMut-for-PathBuf)

These APIs are now stable in const contexts:

- [`VecDeque::new`](https://doc.rust-lang.org/stable/std/collections/struct.VecDeque.html#method.new)

<a id="1.68.0-Cargo"></a>

Cargo
-----

- [Stabilize sparse registry support for crates.io](https://github.com/rust-lang/cargo/pull/11224/)
- [`cargo build --verbose` tells you more about why it recompiles.](https://github.com/rust-lang/cargo/pull/11407/)
- [Show progress of crates.io index update even `net.git-fetch-with-cli` option enabled](https://github.com/rust-lang/cargo/pull/11579/)

<a id="1.68.0-Misc"></a>

Misc
----

<a id="1.68.0-Compatibility-Notes"></a>

Compatibility Notes
-------------------

- [Only support Android NDK 25 or newer](https://blog.rust-lang.org/2023/01/09/android-ndk-update-r25.html)
- [Add `SEMICOLON_IN_EXPRESSIONS_FROM_MACROS` to future-incompat report](https://github.com/rust-lang/rust/pull/103418/)
- [Only specify `--target` by default for `-Zgcc-ld=lld` on wasm](https://github.com/rust-lang/rust/pull/101792/)
- [Bump `IMPLIED_BOUNDS_ENTAILMENT` to Deny + ReportNow](https://github.com/rust-lang/rust/pull/106465/)
- [`std::task::Context` no longer implements Send and Sync](https://github.com/rust-lang/rust/pull/95985)

<a id="1.68.0-Internal-Changes"></a>

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.

- [Encode spans relative to the enclosing item](https://github.com/rust-lang/rust/pull/84762/)
- [Don't normalize in AstConv](https://github.com/rust-lang/rust/pull/101947/)
- [Find the right lower bound region in the scenario of partial order relations](https://github.com/rust-lang/rust/pull/104765/)
- [Fix impl block in const expr](https://github.com/rust-lang/rust/pull/104889/)
- [Check ADT fields for copy implementations considering regions](https://github.com/rust-lang/rust/pull/105102/)
- [rustdoc: simplify JS search routine by not messing with lev distance](https://github.com/rust-lang/rust/pull/105796/)
- [Enable ThinLTO for rustc on `x86_64-pc-windows-msvc`](https://github.com/rust-lang/rust/pull/103591/)
- [Enable ThinLTO for rustc on `x86_64-apple-darwin`](https://github.com/rust-lang/rust/pull/103647/)

Version 1.67.1 (2023-02-09)
===========================

- [Fix interoperability with thin archives.](https://github.com/rust-lang/rust/pull/107360)
- [Fix an internal error in the compiler build process.](https://github.com/rust-lang/rust/pull/105624)
- [Downgrade `clippy::uninlined_format_args` to pedantic.](https://github.com/rust-lang/rust-clippy/pull/10265)

Version 1.67.0 (2023-01-26)
==========================

<a id="1.67.0-Language"></a>

Language
--------

- [Make `Sized` predicates coinductive, allowing cycles.](https://github.com/rust-lang/rust/pull/100386/)
- [`#[must_use]` annotations on `async fn` also affect the `Future::Output`.](https://github.com/rust-lang/rust/pull/100633/)
- [Elaborate supertrait obligations when deducing closure signatures.](https://github.com/rust-lang/rust/pull/101834/)
- [Invalid literals are no longer an error under `cfg(FALSE)`.](https://github.com/rust-lang/rust/pull/102944/)
- [Unreserve braced enum variants in value namespace.](https://github.com/rust-lang/rust/pull/103578/)

<a id="1.67.0-Compiler"></a>

Compiler
--------

- [Enable varargs support for calling conventions other than `C` or `cdecl`.](https://github.com/rust-lang/rust/pull/97971/)
- [Add new MIR constant propagation based on dataflow analysis.](https://github.com/rust-lang/rust/pull/101168/)
- [Optimize field ordering by grouping m\*2^n-sized fields with equivalently aligned ones.](https://github.com/rust-lang/rust/pull/102750/)
- [Stabilize native library modifier `verbatim`.](https://github.com/rust-lang/rust/pull/104360/)

Added, updated, and removed targets:

- [Add a tier 3 target for PowerPC on AIX](https://github.com/rust-lang/rust/pull/102293/), `powerpc64-ibm-aix`.
- [Add a tier 3 target for the Sony PlayStation 1](https://github.com/rust-lang/rust/pull/102689/), `mipsel-sony-psx`.
- [Add tier 3 `no_std` targets for the QNX Neutrino RTOS](https://github.com/rust-lang/rust/pull/102701/),
  `aarch64-unknown-nto-qnx710` and `x86_64-pc-nto-qnx710`.
- [Promote UEFI targets to tier 2](https://github.com/rust-lang/rust/pull/103933/), `aarch64-unknown-uefi`, `i686-unknown-uefi`, and `x86_64-unknown-uefi`.
- [Remove tier 3 `linuxkernel` targets](https://github.com/rust-lang/rust/pull/104015/) (not used by the actual kernel).

Refer to Rust's [platform support page][platform-support-doc]
for more information on Rust's tiered platform support.

<a id="1.67.0-Libraries"></a>

Libraries
---------

- [Merge `crossbeam-channel` into `std::sync::mpsc`.](https://github.com/rust-lang/rust/pull/93563/)
- [Fix inconsistent rounding of 0.5 when formatted to 0 decimal places.](https://github.com/rust-lang/rust/pull/102935/)
- [Derive `Eq` and `Hash` for `ControlFlow`.](https://github.com/rust-lang/rust/pull/103084/)
- [Don't build `compiler_builtins` with `-C panic=abort`.](https://github.com/rust-lang/rust/pull/103786/)

<a id="1.67.0-Stabilized-APIs"></a>

Stabilized APIs
---------------

- [`{integer}::checked_ilog`](https://doc.rust-lang.org/stable/std/primitive.i32.html#method.checked_ilog)
- [`{integer}::checked_ilog2`](https://doc.rust-lang.org/stable/std/primitive.i32.html#method.checked_ilog2)
- [`{integer}::checked_ilog10`](https://doc.rust-lang.org/stable/std/primitive.i32.html#method.checked_ilog10)
- [`{integer}::ilog`](https://doc.rust-lang.org/stable/std/primitive.i32.html#method.ilog)
- [`{integer}::ilog2`](https://doc.rust-lang.org/stable/std/primitive.i32.html#method.ilog2)
- [`{integer}::ilog10`](https://doc.rust-lang.org/stable/std/primitive.i32.html#method.ilog10)
- [`NonZeroU*::ilog2`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroU32.html#method.ilog2)
- [`NonZeroU*::ilog10`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroU32.html#method.ilog10)
- [`NonZero*::BITS`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroU32.html#associatedconstant.BITS)

These APIs are now stable in const contexts:

- [`char::from_u32`](https://doc.rust-lang.org/stable/std/primitive.char.html#method.from_u32)
- [`char::from_digit`](https://doc.rust-lang.org/stable/std/primitive.char.html#method.from_digit)
- [`char::to_digit`](https://doc.rust-lang.org/stable/std/primitive.char.html#method.to_digit)
- [`core::char::from_u32`](https://doc.rust-lang.org/stable/core/char/fn.from_u32.html)
- [`core::char::from_digit`](https://doc.rust-lang.org/stable/core/char/fn.from_digit.html)

<a id="1.67.0-Compatibility-Notes"></a>

Compatibility Notes
-------------------

- [The layout of `repr(Rust)` types now groups m\*2^n-sized fields with
  equivalently aligned ones.](https://github.com/rust-lang/rust/pull/102750/)
  This is intended to be an optimization, but it is also known to increase type
  sizes in a few cases for the placement of enum tags. As a reminder, the layout
  of `repr(Rust)` types is an implementation detail, subject to change.
- [0.5 now rounds to 0 when formatted to 0 decimal places.](https://github.com/rust-lang/rust/pull/102935/)
  This makes it consistent with the rest of floating point formatting that
  rounds ties toward even digits.
- [Chains of `&&` and `||` will now drop temporaries from their sub-expressions in
  evaluation order, left-to-right.](https://github.com/rust-lang/rust/pull/103293/)
  Previously, it was "twisted" such that the _first_ expression dropped its
  temporaries _last_, after all of the other expressions dropped in order.
- [Underscore suffixes on string literals are now a hard error.](https://github.com/rust-lang/rust/pull/103914/)
  This has been a future-compatibility warning since 1.20.0.
- [Stop passing `-export-dynamic` to `wasm-ld`.](https://github.com/rust-lang/rust/pull/105405/)
- [`main` is now mangled as `__main_void` on `wasm32-wasi`.](https://github.com/rust-lang/rust/pull/105468/)
- [Cargo now emits an error if there are multiple registries in the configuration
  with the same index URL.](https://github.com/rust-lang/cargo/pull/10592)

<a id="1.67.0-Internal-Changes"></a>

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.

- [Rewrite LLVM's archive writer in Rust.](https://github.com/rust-lang/rust/pull/97485/)

Version 1.66.1 (2023-01-10)
===========================

- Added validation of SSH host keys for git URLs in Cargo ([CVE-2022-46176](https://www.cve.org/CVERecord?id=CVE-2022-46176))

Version 1.66.0 (2022-12-15)
==========================

Language
--------
- [Permit specifying explicit discriminants on all `repr(Int)` enums](https://github.com/rust-lang/rust/pull/95710/)
  ```rust
  #[repr(u8)]
  enum Foo {
      A(u8) = 0,
      B(i8) = 1,
      C(bool) = 42,
  }
  ```
- [Allow transmutes between the same type differing only in lifetimes](https://github.com/rust-lang/rust/pull/101520/)
- [Change constant evaluation errors from a deny-by-default lint to a hard error](https://github.com/rust-lang/rust/pull/102091/)
- [Trigger `must_use` on `impl Trait` for supertraits](https://github.com/rust-lang/rust/pull/102287/)
  This makes `impl ExactSizeIterator` respect the existing `#[must_use]` annotation on `Iterator`.
- [Allow `..=X` in patterns](https://github.com/rust-lang/rust/pull/102275/)
- [Uplift `clippy::for_loops_over_fallibles` lint into rustc](https://github.com/rust-lang/rust/pull/99696/)
- [Stabilize `sym` operands in inline assembly](https://github.com/rust-lang/rust/pull/103168/)
- [Update to Unicode 15](https://github.com/rust-lang/rust/pull/101912/)
- [Opaque types no longer imply lifetime bounds](https://github.com/rust-lang/rust/pull/95474/)
  This is a soundness fix which may break code that was erroneously relying on this behavior.

Compiler
--------
- [Add armv5te-none-eabi and thumbv5te-none-eabi tier 3 targets](https://github.com/rust-lang/rust/pull/101329/)
  - Refer to Rust's [platform support page][platform-support-doc] for more
    information on Rust's tiered platform support.
- [Add support for linking against macOS universal libraries](https://github.com/rust-lang/rust/pull/98736)

Libraries
---------
- [Fix `#[derive(Default)]` on a generic `#[default]` enum adding unnecessary `Default` bounds](https://github.com/rust-lang/rust/pull/101040/)
- [Update to Unicode 15](https://github.com/rust-lang/rust/pull/101821/)

Stabilized APIs
---------------

- [`proc_macro::Span::source_text`](https://doc.rust-lang.org/stable/proc_macro/struct.Span.html#method.source_text)
- [`uX::{checked_add_signed, overflowing_add_signed, saturating_add_signed, wrapping_add_signed}`](https://doc.rust-lang.org/stable/std/primitive.u8.html#method.checked_add_signed)
- [`iX::{checked_add_unsigned, overflowing_add_unsigned, saturating_add_unsigned, wrapping_add_unsigned}`](https://doc.rust-lang.org/stable/std/primitive.i8.html#method.checked_add_unsigned)
- [`iX::{checked_sub_unsigned, overflowing_sub_unsigned, saturating_sub_unsigned, wrapping_sub_unsigned}`](https://doc.rust-lang.org/stable/std/primitive.i8.html#method.checked_sub_unsigned)
- [`BTreeSet::{first, last, pop_first, pop_last}`](https://doc.rust-lang.org/stable/std/collections/struct.BTreeSet.html#method.first)
- [`BTreeMap::{first_key_value, last_key_value, first_entry, last_entry, pop_first, pop_last}`](https://doc.rust-lang.org/stable/std/collections/struct.BTreeMap.html#method.first_key_value)
- [Add `AsFd` implementations for stdio lock types on WASI.](https://github.com/rust-lang/rust/pull/101768/)
- [`impl TryFrom<Vec<T>> for Box<[T; N]>`](https://doc.rust-lang.org/stable/std/boxed/struct.Box.html#impl-TryFrom%3CVec%3CT%2C%20Global%3E%3E-for-Box%3C%5BT%3B%20N%5D%2C%20Global%3E)
- [`core::hint::black_box`](https://doc.rust-lang.org/stable/std/hint/fn.black_box.html)
- [`Duration::try_from_secs_{f32,f64}`](https://doc.rust-lang.org/stable/std/time/struct.Duration.html#method.try_from_secs_f32)
- [`Option::unzip`](https://doc.rust-lang.org/stable/std/option/enum.Option.html#method.unzip)
- [`std::os::fd`](https://doc.rust-lang.org/stable/std/os/fd/index.html)


Rustdoc
-------

- [Add Rustdoc warning for invalid HTML tags in the documentation](https://github.com/rust-lang/rust/pull/101720/)

Cargo
-----

- [Added `cargo remove` to remove dependencies from Cargo.toml](https://doc.rust-lang.org/nightly/cargo/commands/cargo-remove.html)
- [`cargo publish` now waits for the new version to be downloadable before exiting](https://github.com/rust-lang/cargo/pull/11062)

See [detailed release notes](https://github.com/rust-lang/cargo/blob/master/CHANGELOG.md#cargo-166-2022-12-15) for more.

Compatibility Notes
-------------------

- [Only apply `ProceduralMasquerade` hack to older versions of `rental`](https://github.com/rust-lang/rust/pull/94063/)
- [Don't export `__heap_base` and `__data_end` on wasm32-wasi.](https://github.com/rust-lang/rust/pull/102385/)
- [Don't export `__wasm_init_memory` on WebAssembly.](https://github.com/rust-lang/rust/pull/102426/)
- [Only export `__tls_*` on wasm32-unknown-unknown.](https://github.com/rust-lang/rust/pull/102440/)
- [Don't link to `libresolv` in libstd on Darwin](https://github.com/rust-lang/rust/pull/102766/)
- [Update libstd's libc to 0.2.135 (to make `libstd` no longer pull in `libiconv.dylib` on Darwin)](https://github.com/rust-lang/rust/pull/103277/)
- [Opaque types no longer imply lifetime bounds](https://github.com/rust-lang/rust/pull/95474/)
  This is a soundness fix which may break code that was erroneously relying on this behavior.
- [Make `order_dependent_trait_objects` show up in future-breakage reports](https://github.com/rust-lang/rust/pull/102635/)
- [Change std::process::Command spawning to default to inheriting the parent's signal mask](https://github.com/rust-lang/rust/pull/101077/)

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.

- [Enable BOLT for LLVM compilation](https://github.com/rust-lang/rust/pull/94381/)
- [Enable LTO for rustc_driver.so](https://github.com/rust-lang/rust/pull/101403/)

Version 1.65.0 (2022-11-03)
==========================

Language
--------
- [Error on `as` casts of enums with `#[non_exhaustive]` variants](https://github.com/rust-lang/rust/pull/92744/)
- [Stabilize `let else`](https://github.com/rust-lang/rust/pull/93628/)
- [Stabilize generic associated types (GATs)](https://github.com/rust-lang/rust/pull/96709/)
- [Add lints `let_underscore_drop` and `let_underscore_lock` from Clippy](https://github.com/rust-lang/rust/pull/97739/)
- [Stabilize `break`ing from arbitrary labeled blocks ("label-break-value")](https://github.com/rust-lang/rust/pull/99332/)
- [Uninitialized integers, floats, and raw pointers are now considered immediate UB](https://github.com/rust-lang/rust/pull/98919/).
  Usage of `MaybeUninit` is the correct way to work with uninitialized memory.
- [Stabilize raw-dylib for Windows x86_64, aarch64, and thumbv7a](https://github.com/rust-lang/rust/pull/99916/)
- [Do not allow `Drop` impl on foreign ADTs](https://github.com/rust-lang/rust/pull/99576/)

Compiler
--------
- [Stabilize -Csplit-debuginfo on Linux](https://github.com/rust-lang/rust/pull/98051/)
- [Use niche-filling optimization even when multiple variants have data](https://github.com/rust-lang/rust/pull/94075/)
- [Associated type projections are now verified to be well-formed prior to resolving the underlying type](https://github.com/rust-lang/rust/pull/99217/#issuecomment-1209365630)
- [Stringify non-shorthand visibility correctly](https://github.com/rust-lang/rust/pull/100350/)
- [Normalize struct field types when unsizing](https://github.com/rust-lang/rust/pull/101831/)
- [Update to LLVM 15](https://github.com/rust-lang/rust/pull/99464/)
- [Fix aarch64 call abi to correctly zeroext when needed](https://github.com/rust-lang/rust/pull/97800/)
- [debuginfo: Generalize C++-like encoding for enums](https://github.com/rust-lang/rust/pull/98393/)
- [Add `special_module_name` lint](https://github.com/rust-lang/rust/pull/94467/)
- [Add support for generating unique profraw files by default when using `-C instrument-coverage`](https://github.com/rust-lang/rust/pull/100384/)
- [Allow dynamic linking for iOS/tvOS targets](https://github.com/rust-lang/rust/pull/100636/)

New targets:

- [Add armv4t-none-eabi as a tier 3 target](https://github.com/rust-lang/rust/pull/100244/)
- [Add powerpc64-unknown-openbsd and riscv64-unknown-openbsd as tier 3 targets](https://github.com/rust-lang/rust/pull/101025/)
  - Refer to Rust's [platform support page][platform-support-doc] for more
    information on Rust's tiered platform support.

Libraries
---------

- [Don't generate `PartialEq::ne` in derive(PartialEq)](https://github.com/rust-lang/rust/pull/98655/)
- [Windows RNG: Use `BCRYPT_RNG_ALG_HANDLE` by default](https://github.com/rust-lang/rust/pull/101325/)
- [Forbid mixing `System` with direct system allocator calls](https://github.com/rust-lang/rust/pull/101394/)
- [Document no support for writing to non-blocking stdio/stderr](https://github.com/rust-lang/rust/pull/101416/)
- [`std::layout::Layout` size must not overflow `isize::MAX` when rounded up to `align`](https://github.com/rust-lang/rust/pull/95295)
  This also changes the safety conditions on `Layout::from_size_align_unchecked`.

Stabilized APIs
---------------

- [`std::backtrace::Backtrace`](https://doc.rust-lang.org/stable/std/backtrace/struct.Backtrace.html)
- [`Bound::as_ref`](https://doc.rust-lang.org/stable/std/ops/enum.Bound.html#method.as_ref)
- [`std::io::read_to_string`](https://doc.rust-lang.org/stable/std/io/fn.read_to_string.html)
- [`<*const T>::cast_mut`](https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.cast_mut)
- [`<*mut T>::cast_const`](https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.cast_const)

These APIs are now stable in const contexts:

- [`<*const T>::offset_from`](https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset_from)
- [`<*mut T>::offset_from`](https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset_from)

Cargo
-----

- [Apply GitHub fast path even for partial hashes](https://github.com/rust-lang/cargo/pull/10807/)
- [Do not add home bin path to PATH if it's already there](https://github.com/rust-lang/cargo/pull/11023/)
- [Take priority into account within the pending queue](https://github.com/rust-lang/cargo/pull/11032/).
  This slightly optimizes job scheduling by Cargo, with typically small improvements on larger crate graph builds.

Compatibility Notes
-------------------

- [`std::layout::Layout` size must not overflow `isize::MAX` when rounded up to `align`](https://github.com/rust-lang/rust/pull/95295).
  This also changes the safety conditions on `Layout::from_size_align_unchecked`.
- [`PollFn` now only implements `Unpin` if the closure is `Unpin`](https://github.com/rust-lang/rust/pull/102737).
  This is a possible breaking change if users were relying on the blanket unpin implementation.
  See discussion on the PR for details of why this change was made.
- [Drop ExactSizeIterator impl from std::char::EscapeAscii](https://github.com/rust-lang/rust/pull/99880)
  This is a backwards-incompatible change to the standard library's surface
  area, but is unlikely to affect real world usage.
- [Do not consider a single repeated lifetime eligible for elision in the return type](https://github.com/rust-lang/rust/pull/103450)
  This behavior was unintentionally changed in 1.64.0, and this release reverts that change by making this an error again.
- [Reenable disabled early syntax gates as future-incompatibility lints](https://github.com/rust-lang/rust/pull/99935/)
- [Update the minimum external LLVM to 13](https://github.com/rust-lang/rust/pull/100460/)
- [Don't duplicate file descriptors into stdio fds](https://github.com/rust-lang/rust/pull/101426/)
- [Sunset RLS](https://github.com/rust-lang/rust/pull/100863/)
- [Deny usage of `#![cfg_attr(..., crate_type = ...)]` to set the crate type](https://github.com/rust-lang/rust/pull/99784/)
  This strengthens the forward compatibility lint deprecated_cfg_attr_crate_type_name to deny.
- [`llvm-has-rust-patches` allows setting the build system to treat the LLVM as having Rust-specific patches](https://github.com/rust-lang/rust/pull/101072)
  This option may need to be set for distributions that are building Rust with a patched LLVM via `llvm-config`, not the built-in LLVM.
- Combining three or more languages (e.g. Objective C, C++ and Rust) into one binary may hit linker limitations when using `lld`. For more information, see [issue 102754][102754].

[102754]: https://github.com/rust-lang/rust/issues/102754

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.

- [Add `x.sh` and `x.ps1` shell scripts](https://github.com/rust-lang/rust/pull/99992/)
- [compiletest: use target cfg instead of hard-coded tables](https://github.com/rust-lang/rust/pull/100260/)
- [Use object instead of LLVM for reading bitcode from rlibs](https://github.com/rust-lang/rust/pull/98100/)
- [Enable MIR inlining for optimized compilations](https://github.com/rust-lang/rust/pull/91743)
  This provides a 3-10% improvement in compiletimes for real world crates. See [perf results](https://perf.rust-lang.org/compare.html?start=aedf78e56b2279cc869962feac5153b6ba7001ed&end=0075bb4fad68e64b6d1be06bf2db366c30bc75e1&stat=instructions:u).

Version 1.64.0 (2022-09-22)
===========================

Language
--------
- [Unions with mutable references or tuples of allowed types are now allowed](https://github.com/rust-lang/rust/pull/97995/)
- It is now considered valid to deallocate memory pointed to by a shared reference `&T` [if every byte in `T` is inside an `UnsafeCell`](https://github.com/rust-lang/rust/pull/98017/)
- Unused tuple struct fields are now warned against in an allow-by-default lint, [`unused_tuple_struct_fields`](https://github.com/rust-lang/rust/pull/95977/), similar to the existing warning for unused struct fields. This lint will become warn-by-default in the future.

Compiler
--------
- [Add Nintendo Switch as tier 3 target](https://github.com/rust-lang/rust/pull/88991/)
  - Refer to Rust's [platform support page][platform-support-doc] for more
    information on Rust's tiered platform support.
- [Only compile `#[used]` as llvm.compiler.used for ELF targets](https://github.com/rust-lang/rust/pull/93718/)
- [Add the `--diagnostic-width` compiler flag to define the terminal width.](https://github.com/rust-lang/rust/pull/95635/)
- [Add support for link-flavor `rust-lld` for iOS, tvOS and watchOS](https://github.com/rust-lang/rust/pull/98771/)

Libraries
---------
- [Remove restrictions on compare-exchange memory ordering.](https://github.com/rust-lang/rust/pull/98383/)
- You can now `write!` or `writeln!` into an `OsString`: [Implement `fmt::Write` for `OsString`](https://github.com/rust-lang/rust/pull/97915/)
- [Make RwLockReadGuard covariant](https://github.com/rust-lang/rust/pull/96820/)
- [Implement `FusedIterator` for `std::net::[Into]Incoming`](https://github.com/rust-lang/rust/pull/97300/)
- [`impl<T: AsRawFd> AsRawFd for {Arc,Box}<T>`](https://github.com/rust-lang/rust/pull/97437/)
- [`ptr::copy` and `ptr::swap` are doing untyped copies](https://github.com/rust-lang/rust/pull/97712/)
- [Add cgroupv1 support to `available_parallelism`](https://github.com/rust-lang/rust/pull/97925/)
- [Mitigate many incorrect uses of `mem::uninitialized`](https://github.com/rust-lang/rust/pull/99182/)

Stabilized APIs
---------------

- [`future::IntoFuture`](https://doc.rust-lang.org/stable/std/future/trait.IntoFuture.html)
- [`future::poll_fn`](https://doc.rust-lang.org/stable/std/future/fn.poll_fn.html)
- [`task::ready!`](https://doc.rust-lang.org/stable/std/task/macro.ready.html)
- [`num::NonZero*::checked_mul`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroUsize.html#method.checked_mul)
- [`num::NonZero*::checked_pow`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroUsize.html#method.checked_pow)
- [`num::NonZero*::saturating_mul`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroUsize.html#method.saturating_mul)
- [`num::NonZero*::saturating_pow`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroUsize.html#method.saturating_pow)
- [`num::NonZeroI*::abs`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroIsize.html#method.abs)
- [`num::NonZeroI*::checked_abs`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroIsize.html#method.checked_abs)
- [`num::NonZeroI*::overflowing_abs`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroIsize.html#method.overflowing_abs)
- [`num::NonZeroI*::saturating_abs`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroIsize.html#method.saturating_abs)
- [`num::NonZeroI*::unsigned_abs`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroIsize.html#method.unsigned_abs)
- [`num::NonZeroI*::wrapping_abs`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroIsize.html#method.wrapping_abs)
- [`num::NonZeroU*::checked_add`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroUsize.html#method.checked_add)
- [`num::NonZeroU*::checked_next_power_of_two`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroUsize.html#method.checked_next_power_of_two)
- [`num::NonZeroU*::saturating_add`](https://doc.rust-lang.org/stable/std/num/struct.NonZeroUsize.html#method.saturating_add)
- [`os::unix::process::CommandExt::process_group`](https://doc.rust-lang.org/stable/std/os/unix/process/trait.CommandExt.html#tymethod.process_group)
- [`os::windows::fs::FileTypeExt::is_symlink_dir`](https://doc.rust-lang.org/stable/std/os/windows/fs/trait.FileTypeExt.html#tymethod.is_symlink_dir)
- [`os::windows::fs::FileTypeExt::is_symlink_file`](https://doc.rust-lang.org/stable/std/os/windows/fs/trait.FileTypeExt.html#tymethod.is_symlink_file)

These types were previously stable in `std::ffi`, but are now also available in `core` and `alloc`:

- [`core::ffi::CStr`](https://doc.rust-lang.org/stable/core/ffi/struct.CStr.html)
- [`core::ffi::FromBytesWithNulError`](https://doc.rust-lang.org/stable/core/ffi/struct.FromBytesWithNulError.html)
- [`alloc::ffi::CString`](https://doc.rust-lang.org/stable/alloc/ffi/struct.CString.html)
- [`alloc::ffi::FromVecWithNulError`](https://doc.rust-lang.org/stable/alloc/ffi/struct.FromVecWithNulError.html)
- [`alloc::ffi::IntoStringError`](https://doc.rust-lang.org/stable/alloc/ffi/struct.IntoStringError.html)
- [`alloc::ffi::NulError`](https://doc.rust-lang.org/stable/alloc/ffi/struct.NulError.html)

These types were previously stable in `std::os::raw`, but are now also available in `core::ffi` and `std::ffi`:

- [`ffi::c_char`](https://doc.rust-lang.org/stable/std/ffi/type.c_char.html)
- [`ffi::c_double`](https://doc.rust-lang.org/stable/std/ffi/type.c_double.html)
- [`ffi::c_float`](https://doc.rust-lang.org/stable/std/ffi/type.c_float.html)
- [`ffi::c_int`](https://doc.rust-lang.org/stable/std/ffi/type.c_int.html)
- [`ffi::c_long`](https://doc.rust-lang.org/stable/std/ffi/type.c_long.html)
- [`ffi::c_longlong`](https://doc.rust-lang.org/stable/std/ffi/type.c_longlong.html)
- [`ffi::c_schar`](https://doc.rust-lang.org/stable/std/ffi/type.c_schar.html)
- [`ffi::c_short`](https://doc.rust-lang.org/stable/std/ffi/type.c_short.html)
- [`ffi::c_uchar`](https://doc.rust-lang.org/stable/std/ffi/type.c_uchar.html)
- [`ffi::c_uint`](https://doc.rust-lang.org/stable/std/ffi/type.c_uint.html)
- [`ffi::c_ulong`](https://doc.rust-lang.org/stable/std/ffi/type.c_ulong.html)
- [`ffi::c_ulonglong`](https://doc.rust-lang.org/stable/std/ffi/type.c_ulonglong.html)
- [`ffi::c_ushort`](https://doc.rust-lang.org/stable/std/ffi/type.c_ushort.html)

These APIs are now usable in const contexts:

- [`slice::from_raw_parts`](https://doc.rust-lang.org/stable/core/slice/fn.from_raw_parts.html)

Cargo
-----
- [Packages can now inherit settings from the workspace so that the settings
  can be centralized in one place.](https://github.com/rust-lang/cargo/pull/10859) See
  [`workspace.package`](https://doc.rust-lang.org/nightly/cargo/reference/workspaces.html#the-workspacepackage-table)
  and
  [`workspace.dependencies`](https://doc.rust-lang.org/nightly/cargo/reference/workspaces.html#the-workspacedependencies-table)
  for more details on how to define these common settings.
- [Cargo commands can now accept multiple `--target` flags to build for
  multiple targets at once](https://github.com/rust-lang/cargo/pull/10766), and the
  [`build.target`](https://doc.rust-lang.org/nightly/cargo/reference/config.html#buildtarget)
  config option may now take an array of multiple targets.
- [The `--jobs` argument can now take a negative number to count backwards from
  the max CPUs.](https://github.com/rust-lang/cargo/pull/10844)
- [`cargo add` will now update `Cargo.lock`.](https://github.com/rust-lang/cargo/pull/10902)
- [Added](https://github.com/rust-lang/cargo/pull/10838) the
  [`--crate-type`](https://doc.rust-lang.org/nightly/cargo/commands/cargo-rustc.html#option-cargo-rustc---crate-type)
  flag to `cargo rustc` to override the crate type.
- [Significantly improved the performance fetching git dependencies from GitHub
  when using a hash in the `rev` field.](https://github.com/rust-lang/cargo/pull/10079)

Misc
----
- [The `rust-analyzer` rustup component is now available on the stable channel.](https://github.com/rust-lang/rust/pull/98640/)

Compatibility Notes
-------------------
- The minimum required versions for all `-linux-gnu` targets are now at least kernel 3.2 and glibc 2.17, for targets that previously supported older versions: [Increase the minimum linux-gnu versions](https://github.com/rust-lang/rust/pull/95026/)
- [Network primitives are now implemented with the ideal Rust layout, not the C system layout](https://github.com/rust-lang/rust/pull/78802/). This can cause problems when transmuting the types.
- [Add assertion that `transmute_copy`'s `U` is not larger than `T`](https://github.com/rust-lang/rust/pull/98839/)
- [A soundness bug in `BTreeMap` was fixed](https://github.com/rust-lang/rust/pull/99413/) that allowed data it was borrowing to be dropped before the container.
- [The Drop behavior of C-like enums cast to ints has changed](https://github.com/rust-lang/rust/pull/96862/). These are already discouraged by a compiler warning.
- [Relate late-bound closure lifetimes to parent fn in NLL](https://github.com/rust-lang/rust/pull/98835/)
- [Errors at const-eval time are now in future incompatibility reports](https://github.com/rust-lang/rust/pull/97743/)
- On the `thumbv6m-none-eabi` target, some incorrect `asm!` statements were erroneously accepted if they used the high registers (r8 to r14) as an input/output operand. [This is no longer accepted](https://github.com/rust-lang/rust/pull/99155/).
- [`impl Trait` was accidentally accepted as the associated type value of return-position `impl Trait`](https://github.com/rust-lang/rust/pull/97346/), without fulfilling all the trait bounds of that associated type, as long as the hidden type satisfies said bounds. This has been fixed.

Internal Changes
----------------

These changes do not affect any public interfaces of Rust, but they represent
significant improvements to the performance or internals of rustc and related
tools.

- Windows builds now use profile-guided optimization, providing 10-20% improvements to compiler performance: [Utilize PGO for windows x64 rustc dist builds](https://github.com/rust-lang/rust/pull/96978/)
- [Stop keeping metadata in memory before writing it to disk](https://github.com/rust-lang/rust/pull/96544/)
- [compiletest: strip debuginfo by default for mode=ui](https://github.com/rust-lang/rust/pull/98140/)
- Many improvements to generated code for derives, including performance improvements:
  - [Don't use match-destructuring for derived ops on structs.](https://github.com/rust-lang/rust/pull/98446/)
  - [Many small deriving cleanups](https://github.com/rust-lang/rust/pull/98741/)
  - [More derive output improvements](https://github.com/rust-lang/rust/pull/98758/)
  - [Clarify deriving code](https://github.com/rust-lang/rust/pull/98915/)
  - [Final derive output improvements](https://github.com/rust-lang/rust/pull/99046/)
  - [Stop injecting `#[allow(unused_qualifications)]` in generated `derive` implementations](https://github.com/rust-lang/rust/pull/99485/)
  - [Improve `derive(Debug)`](https://github.com/rust-lang/rust/pull/98190/)
- [Bump to clap 3](https://github.com/rust-lang/rust/pull/98213/)
- [fully move dropck to mir](https://github.com/rust-lang/rust/pull/98641/)
- [Optimize `Vec::insert` for the case where `index == len`.](https://github.com/rust-lang/rust/pull/98755/)
- [Convert rust-analyzer to an in-tree tool](https://github.com/rust-lang/rust/pull/99603/)

Version 1.63.0 (2022-08-11)
==========================

Language
--------
- [Remove migrate borrowck mode for pre-NLL errors.][95565]
- [Modify MIR building to drop repeat expressions with length zero.][95953]
- [Remove label/lifetime shadowing warnings.][96296]
- [Allow explicit generic arguments in the presence of `impl Trait` args.][96868]
- [Make `cenum_impl_drop_cast` warnings deny-by-default.][97652]
- [Prevent unwinding when `-C panic=abort` is used regardless of declared ABI.][96959]
- [lub: don't bail out due to empty binders.][97867]

Compiler
--------
- [Stabilize the `bundle` native library modifier,][95818] also removing the
  deprecated `static-nobundle` linking kind.
- [Add Apple WatchOS compile targets\*.][95243]
- [Add a Windows application manifest to rustc-main.][96737]

\* Refer to Rust's [platform support page][platform-support-doc] for more
   information on Rust's tiered platform support.

Libraries
---------
- [Implement `Copy`, `Clone`, `PartialEq` and `Eq` for `core::fmt::Alignment`.][94530]
- [Extend `ptr::null` and `null_mut` to all thin (including extern) types.][94954]
- [`impl Read and Write for VecDeque<u8>`.][95632]
- [STD support for the Nintendo 3DS.][95897]
- [Use rounding in float to Duration conversion methods.][96051]
- [Make write/print macros eagerly drop temporaries.][96455]
- [Implement internal traits that enable `[OsStr]::join`.][96881]
- [Implement `Hash` for `core::alloc::Layout`.][97034]
- [Add capacity documentation for `OsString`.][97202]
- [Put a bound on collection misbehavior.][97316]
- [Make `std::mem::needs_drop` accept `?Sized`.][97675]
- [`impl Termination for Infallible` and then make the `Result` impls of `Termination` more generic.][97803]
- [Document Rust's stance on `/proc/self/mem`.][97837]

Stabilized APIs
---------------

- [`array::from_fn`]
- [`Box::into_pin`]
- [`BinaryHeap::try_reserve`]
- [`BinaryHeap::try_reserve_exact`]
- [`OsString::try_reserve`]
- [`OsString::try_reserve_exact`]
- [`PathBuf::try_reserve`]
- [`PathBuf::try_reserve_exact`]
- [`Path::try_exists`]
- [`Ref::filter_map`]
- [`RefMut::filter_map`]
- [`NonNull::<[T]>::len`][`NonNull::<slice>::len`]
- [`ToOwned::clone_into`]
- [`Ipv6Addr::to_ipv4_mapped`]
- [`unix::io::AsFd`]
- [`unix::io::BorrowedFd<'fd>`]
- [`unix::io::OwnedFd`]
- [`windows::io::AsHandle`]
- [`windows::io::BorrowedHandle<'handle>`]
- [`windows::io::OwnedHandle`]
- [`windows::io::HandleOrInvalid`]
- [`windows::io::HandleOrNull`]
- [`windows::io::InvalidHandleError`]
- [`windows::io::NullHandleError`]
- [`windows::io::AsSocket`]
- [`windows::io::BorrowedSocket<'handle>`]
- [`windows::io::OwnedSocket`]
- [`thread::scope`]
- [`thread::Scope`]
- [`thread::ScopedJoinHandle`]

These APIs are now usable in const contexts:

- [`array::from_ref`]
- [`slice::from_ref`]
- [`intrinsics::copy`]
- [`intrinsics::copy_nonoverlapping`]
- [`<*const T>::copy_to`]
- [`<*const T>::copy_to_nonoverlapping`]
- [`<*mut T>::copy_to`]
- [`<*mut T>::copy_to_nonoverlapping`]
- [`<*mut T>::copy_from`]
- [`<*mut T>::copy_from_nonoverlapping`]
- [`str::from_utf8`]
- [`Utf8Error::error_len`]
- [`Utf8Error::valid_up_to`]
- [`Condvar::new`]
- [`Mutex::new`]
- [`RwLock::new`]

Cargo
-----
- [Stabilize the `--config path` command-line argument.][cargo/10755]
- [Expose rust-version in the environment as `CARGO_PKG_RUST_VERSION`.][cargo/10713]

Compatibility Notes
-------------------

- [`#[link]` attributes are now checked more strictly,][96885] which may introduce
  errors for invalid attribute arguments that were previously ignored.
- [Rounding is now used when converting a float to a `Duration`.][96051] The converted
  duration can differ slightly from what it was.

Internal Changes
----------------

These changes provide no direct user facing benefits, but represent significant
improvements to the internals and overall performance of rustc
and related tools.

- [Prepare Rust for LLVM opaque pointers.][94214]

[94214]: https://github.com/rust-lang/rust/pull/94214/
[94530]: https://github.com/rust-lang/rust/pull/94530/
[94954]: https://github.com/rust-lang/rust/pull/94954/
[95243]: https://github.com/rust-lang/rust/pull/95243/
[95565]: https://github.com/rust-lang/rust/pull/95565/
[95632]: https://github.com/rust-lang/rust/pull/95632/
[95818]: https://github.com/rust-lang/rust/pull/95818/
[95897]: https://github.com/rust-lang/rust/pull/95897/
[95953]: https://github.com/rust-lang/rust/pull/95953/
[96051]: https://github.com/rust-lang/rust/pull/96051/
[96296]: https://github.com/rust-lang/rust/pull/96296/
[96455]: https://github.com/rust-lang/rust/pull/96455/
[96737]: https://github.com/rust-lang/rust/pull/96737/
[96868]: https://github.com/rust-lang/rust/pull/96868/
[96881]: https://github.com/rust-lang/rust/pull/96881/
[96885]: https://github.com/rust-lang/rust/pull/96885/
[96959]: https://github.com/rust-lang/rust/pull/96959/
[97034]: https://github.com/rust-lang/rust/pull/97034/
[97202]: https://github.com/rust-lang/rust/pull/97202/
[97316]: https://github.com/rust-lang/rust/pull/97316/
[97652]: https://github.com/rust-lang/rust/pull/97652/
[97675]: https://github.com/rust-lang/rust/pull/97675/
[97803]: https://github.com/rust-lang/rust/pull/97803/
[97837]: https://github.com/rust-lang/rust/pull/97837/
[97867]: https://github.com/rust-lang/rust/pull/97867/
[cargo/10713]: https://github.com/rust-lang/cargo/pull/10713/
[cargo/10755]: https://github.com/rust-lang/cargo/pull/10755/

[`array::from_fn`]: https://doc.rust-lang.org/stable/std/array/fn.from_fn.html
[`Box::into_pin`]: https://doc.rust-lang.org/stable/std/boxed/struct.Box.html#method.into_pin
[`BinaryHeap::try_reserve_exact`]: https://doc.rust-lang.org/stable/alloc/collections/binary_heap/struct.BinaryHeap.html#method.try_reserve_exact
[`BinaryHeap::try_reserve`]: https://doc.rust-lang.org/stable/std/collections/struct.BinaryHeap.html#method.try_reserve
[`OsString::try_reserve`]: https://doc.rust-lang.org/stable/std/ffi/struct.OsString.html#method.try_reserve
[`OsString::try_reserve_exact`]: https://doc.rust-lang.org/stable/std/ffi/struct.OsString.html#method.try_reserve_exact
[`PathBuf::try_reserve`]: https://doc.rust-lang.org/stable/std/path/struct.PathBuf.html#method.try_reserve
[`PathBuf::try_reserve_exact`]: https://doc.rust-lang.org/stable/std/path/struct.PathBuf.html#method.try_reserve_exact
[`Path::try_exists`]: https://doc.rust-lang.org/stable/std/path/struct.Path.html#method.try_exists
[`Ref::filter_map`]: https://doc.rust-lang.org/stable/std/cell/struct.Ref.html#method.filter_map
[`RefMut::filter_map`]: https://doc.rust-lang.org/stable/std/cell/struct.RefMut.html#method.filter_map
[`NonNull::<slice>::len`]: https://doc.rust-lang.org/stable/std/ptr/struct.NonNull.html#method.len
[`ToOwned::clone_into`]: https://doc.rust-lang.org/stable/std/borrow/trait.ToOwned.html#method.clone_into
[`Ipv6Addr::to_ipv4_mapped`]: https://doc.rust-lang.org/stable/std/net/struct.Ipv6Addr.html#method.to_ipv4_mapped
[`unix::io::AsFd`]: https://doc.rust-lang.org/stable/std/os/unix/io/trait.AsFd.html
[`unix::io::BorrowedFd<'fd>`]: https://doc.rust-lang.org/stable/std/os/unix/io/struct.BorrowedFd.html
[`unix::io::OwnedFd`]: https://doc.rust-lang.org/stable/std/os/unix/io/struct.OwnedFd.html
[`windows::io::AsHandle`]: https://doc.rust-lang.org/stable/std/os/windows/io/trait.AsHandle.html
[`windows::io::BorrowedHandle<'handle>`]: https://doc.rust-lang.org/stable/std/os/windows/io/struct.BorrowedHandle.html
[`windows::io::OwnedHandle`]: https://doc.rust-lang.org/stable/std/os/windows/io/struct.OwnedHandle.html
[`windows::io::HandleOrInvalid`]: https://doc.rust-lang.org/stable/std/os/windows/io/struct.HandleOrInvalid.html
[`windows::io::HandleOrNull`]: https://doc.rust-lang.org/stable/std/os/windows/io/struct.HandleOrNull.html
[`windows::io::InvalidHandleError`]: https://doc.rust-lang.org/stable/std/os/windows/io/struct.InvalidHandleError.html
[`windows::io::NullHandleError`]: https://doc.rust-lang.org/stable/std/os/windows/io/struct.NullHandleError.html
[`windows::io::AsSocket`]: https://doc.rust-lang.org/stable/std/os/windows/io/trait.AsSocket.html
[`windows::io::BorrowedSocket<'handle>`]: https://doc.rust-lang.org/stable/std/os/windows/io/struct.BorrowedSocket.html
[`windows::io::OwnedSocket`]: https://doc.rust-lang.org/stable/std/os/windows/io/struct.OwnedSocket.html
[`thread::scope`]: https://doc.rust-lang.org/stable/std/thread/fn.scope.html
[`thread::Scope`]: https://doc.rust-lang.org/stable/std/thread/struct.Scope.html
[`thread::ScopedJoinHandle`]: https://doc.rust-lang.org/stable/std/thread/struct.ScopedJoinHandle.html

[`array::from_ref`]: https://doc.rust-lang.org/stable/std/array/fn.from_ref.html
[`slice::from_ref`]: https://doc.rust-lang.org/stable/std/slice/fn.from_ref.html
[`intrinsics::copy`]: https://doc.rust-lang.org/stable/std/intrinsics/fn.copy.html
[`intrinsics::copy_nonoverlapping`]: https://doc.rust-lang.org/stable/std/intrinsics/fn.copy_nonoverlapping.html
[`<*const T>::copy_to`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.copy_to
[`<*const T>::copy_to_nonoverlapping`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.copy_to_nonoverlapping
[`<*mut T>::copy_to`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.copy_to-1
[`<*mut T>::copy_to_nonoverlapping`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.copy_to_nonoverlapping-1
[`<*mut T>::copy_from`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.copy_from
[`<*mut T>::copy_from_nonoverlapping`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.copy_from_nonoverlapping
[`str::from_utf8`]: https://doc.rust-lang.org/stable/std/str/fn.from_utf8.html
[`Utf8Error::error_len`]: https://doc.rust-lang.org/stable/std/str/struct.Utf8Error.html#method.error_len
[`Utf8Error::valid_up_to`]: https://doc.rust-lang.org/stable/std/str/struct.Utf8Error.html#method.valid_up_to
[`Condvar::new`]: https://doc.rust-lang.org/stable/std/sync/struct.Condvar.html#method.new
[`Mutex::new`]: https://doc.rust-lang.org/stable/std/sync/struct.Mutex.html#method.new
[`RwLock::new`]: https://doc.rust-lang.org/stable/std/sync/struct.RwLock.html#method.new

Version 1.62.1 (2022-07-19)
==========================

Rust 1.62.1 addresses a few recent regressions in the compiler and standard
library, and also mitigates a CPU vulnerability on Intel SGX.

* [The compiler fixed unsound function coercions involving `impl Trait` return types.][98608]
* [The compiler fixed an incremental compilation bug with `async fn` lifetimes.][98890]
* [Windows added a fallback for overlapped I/O in synchronous reads and writes.][98950]
* [The `x86_64-fortanix-unknown-sgx` target added a mitigation for the
  MMIO stale data vulnerability][98126], advisory [INTEL-SA-00615].

[98608]: https://github.com/rust-lang/rust/issues/98608
[98890]: https://github.com/rust-lang/rust/issues/98890
[98950]: https://github.com/rust-lang/rust/pull/98950
[98126]: https://github.com/rust-lang/rust/pull/98126
[INTEL-SA-00615]: https://www.intel.com/content/www/us/en/security-center/advisory/intel-sa-00615.html

Version 1.62.0 (2022-06-30)
==========================

Language
--------

- [Stabilize `#[derive(Default)]` on enums with a `#[default]` variant][94457]
- [Teach flow sensitive checks that visibly uninhabited call expressions never return][93313]
- [Fix constants not getting dropped if part of a diverging expression][94775]
- [Support unit struct/enum variant in destructuring assignment][95380]
- [Remove mutable_borrow_reservation_conflict lint and allow the code pattern][96268]
- [`const` functions may now specify `extern "C"` or `extern "Rust"`][95346]

Compiler
--------

- [linker: Stop using whole-archive on dependencies of dylibs][96436]
- [Make `unaligned_references` lint deny-by-default][95372]
  This lint is also a future compatibility lint, and is expected to eventually
  become a hard error.
- [Only add codegen backend to dep info if -Zbinary-dep-depinfo is used][93969]
- [Reject `#[thread_local]` attribute on non-static items][95006]
- [Add tier 3 `aarch64-pc-windows-gnullvm` and `x86_64-pc-windows-gnullvm` targets\*][94872]
- [Implement a lint to warn about unused macro rules][96150]
- [Promote `x86_64-unknown-none` target to Tier 2\*][95705]

\* Refer to Rust's [platform support page][platform-support-doc] for more
   information on Rust's tiered platform support.

Libraries
---------

- [Windows: Use a pipe relay for chaining pipes][95841]
- [Replace Linux Mutex and Condvar with futex based ones.][95035]
- [Replace RwLock by a futex based one on Linux][95801]
- [std: directly use pthread in UNIX parker implementation][96393]

Stabilized APIs
---------------

- [`bool::then_some`]
- [`f32::total_cmp`]
- [`f64::total_cmp`]
- [`Stdin::lines`]
- [`windows::CommandExt::raw_arg`]
- [`impl<T: Default> Default for AssertUnwindSafe<T>`]
- [`From<Rc<str>> for Rc<[u8]>`][rc-u8-from-str]
- [`From<Arc<str>> for Arc<[u8]>`][arc-u8-from-str]
- [`FusedIterator for EncodeWide`]
- [RDM intrinsics on aarch64][stdarch/1285]

Clippy
------

- [Create clippy lint against unexpectedly late drop for temporaries in match scrutinee expressions][94206]

Cargo
-----

- Added the `cargo add` command for adding dependencies to `Cargo.toml` from
  the command-line.
  [docs](https://doc.rust-lang.org/nightly/cargo/commands/cargo-add.html)
- Package ID specs now support `name@version` syntax in addition to the
  previous `name:version` to align with the behavior in `cargo add` and other
  tools. `cargo install` and `cargo yank` also now support this syntax so the
  version does not need to passed as a separate flag.
- The `git` and `registry` directories in Cargo's home directory (usually
  `~/.cargo`) are now marked as cache directories so that they are not
  included in backups or content indexing (on Windows).
- Added automatic `@` argfile support, which will use "response files" if the
  command-line to `rustc` exceeds the operating system's limit.

Compatibility Notes
-------------------

- `cargo test` now passes `--target` to `rustdoc` if the specified target is
  the same as the host target.
  [#10594](https://github.com/rust-lang/cargo/pull/10594)
- [rustdoc: doctests are now run on unexported `macro_rules!` macros, matching other private items][96630]
- [rustdoc: Remove .woff font files][96279]
- [Enforce Copy bounds for repeat elements while considering lifetimes][95819]
- [Windows: Fix potential unsoundness by aborting if `File` reads or writes cannot
  complete synchronously][95469].

Internal Changes
----------------

- [Unify ReentrantMutex implementations across all platforms][96042]

These changes provide no direct user facing benefits, but represent significant
improvements to the internals and overall performance of rustc
and related tools.

[93313]: https://github.com/rust-lang/rust/pull/93313/
[93969]: https://github.com/rust-lang/rust/pull/93969/
[94206]: https://github.com/rust-lang/rust/pull/94206/
[94457]: https://github.com/rust-lang/rust/pull/94457/
[94775]: https://github.com/rust-lang/rust/pull/94775/
[94872]: https://github.com/rust-lang/rust/pull/94872/
[95006]: https://github.com/rust-lang/rust/pull/95006/
[95035]: https://github.com/rust-lang/rust/pull/95035/
[95346]: https://github.com/rust-lang/rust/pull/95346/
[95372]: https://github.com/rust-lang/rust/pull/95372/
[95380]: https://github.com/rust-lang/rust/pull/95380/
[95431]: https://github.com/rust-lang/rust/pull/95431/
[95469]: https://github.com/rust-lang/rust/pull/95469/
[95705]: https://github.com/rust-lang/rust/pull/95705/
[95801]: https://github.com/rust-lang/rust/pull/95801/
[95819]: https://github.com/rust-lang/rust/pull/95819/
[95841]: https://github.com/rust-lang/rust/pull/95841/
[96042]: https://github.com/rust-lang/rust/pull/96042/
[96150]: https://github.com/rust-lang/rust/pull/96150/
[96268]: https://github.com/rust-lang/rust/pull/96268/
[96279]: https://github.com/rust-lang/rust/pull/96279/
[96393]: https://github.com/rust-lang/rust/pull/96393/
[96436]: https://github.com/rust-lang/rust/pull/96436/
[96557]: https://github.com/rust-lang/rust/pull/96557/
[96630]: https://github.com/rust-lang/rust/pull/96630/

[`bool::then_some`]: https://doc.rust-lang.org/stable/std/primitive.bool.html#method.then_some
[`f32::total_cmp`]: https://doc.rust-lang.org/stable/std/primitive.f32.html#method.total_cmp
[`f64::total_cmp`]: https://doc.rust-lang.org/stable/std/primitive.f64.html#method.total_cmp
[`Stdin::lines`]: https://doc.rust-lang.org/stable/std/io/struct.Stdin.html#method.lines
[`impl<T: Default> Default for AssertUnwindSafe<T>`]: https://doc.rust-lang.org/stable/std/panic/struct.AssertUnwindSafe.html#impl-Default
[rc-u8-from-str]: https://doc.rust-lang.org/stable/std/rc/struct.Rc.html#impl-From%3CRc%3Cstr%3E%3E
[arc-u8-from-str]: https://doc.rust-lang.org/stable/std/sync/struct.Arc.html#impl-From%3CArc%3Cstr%3E%3E
[stdarch/1285]: https://github.com/rust-lang/stdarch/pull/1285
[`windows::CommandExt::raw_arg`]: https://doc.rust-lang.org/stable/std/os/windows/process/trait.CommandExt.html#tymethod.raw_arg
[`FusedIterator for EncodeWide`]: https://doc.rust-lang.org/stable/std/os/windows/ffi/struct.EncodeWide.html#impl-FusedIterator

Version 1.61.0 (2022-05-19)
==========================

Language
--------

- [`const fn` signatures can now include generic trait bounds][93827]
- [`const fn` signatures can now use `impl Trait` in argument and return position][93827]
- [Function pointers can now be created, cast, and passed around in a `const fn`][93827]
- [Recursive calls can now set the value of a function's opaque `impl Trait` return type][94081]

Compiler
--------

- [Linking modifier syntax in `#[link]` attributes and on the command line, as well as the `whole-archive` modifier specifically, are now supported][93901]
- [The `char` type is now described as UTF-32 in debuginfo][89887]
- The [`#[target_feature]`][target_feature] attribute [can now be used with aarch64 features][90621]
- X86 [`#[target_feature = "adx"]` is now stable][93745]

Libraries
---------

- [`ManuallyDrop<T>` is now documented to have the same layout as `T`][88375]
- [`#[ignore = "…"]` messages are printed when running tests][92714]
- [Consistently show absent stdio handles on Windows as NULL handles][93263]
- [Make `std::io::stdio::lock()` return `'static` handles.][93965] Previously, the creation of locked handles to stdin/stdout/stderr would borrow the handles being locked, which prevented writing `let out = std::io::stdout().lock();` because `out` would outlive the return value of `stdout()`. Such code now works, eliminating a common pitfall that affected many Rust users.
- [`Vec::from_raw_parts` is now less restrictive about its inputs][95016]
- [`std::thread::available_parallelism` now takes cgroup quotas into account.][92697] Since `available_parallelism` is often used to create a thread pool for parallel computation, which may be CPU-bound for performance, `available_parallelism` will return a value consistent with the ability to use that many threads continuously, if possible. For instance, in a container with 8 virtual CPUs but quotas only allowing for 50% usage, `available_parallelism` will return 4.

Stabilized APIs
---------------

- [`Pin::static_mut`]
- [`Pin::static_ref`]
- [`Vec::retain_mut`]
- [`VecDeque::retain_mut`]
- [`Write` for `Cursor<[u8; N]>`][cursor-write-array]
- [`std::os::unix::net::SocketAddr::from_pathname`]
- [`std::process::ExitCode`] and [`std::process::Termination`]. The stabilization of these two APIs now makes it possible for programs to return errors from `main` with custom exit codes.
- [`std::thread::JoinHandle::is_finished`]

These APIs are now usable in const contexts:

- [`<*const T>::offset` and `<*mut T>::offset`][ptr-offset]
- [`<*const T>::wrapping_offset` and `<*mut T>::wrapping_offset`][ptr-wrapping_offset]
- [`<*const T>::add` and `<*mut T>::add`][ptr-add]
- [`<*const T>::sub` and `<*mut T>::sub`][ptr-sub]
- [`<*const T>::wrapping_add` and `<*mut T>::wrapping_add`][ptr-wrapping_add]
- [`<*const T>::wrapping_sub` and `<*mut T>::wrapping_sub`][ptr-wrapping_sub]
- [`<[T]>::as_mut_ptr`][slice-as_mut_ptr]
- [`<[T]>::as_ptr_range`][slice-as_ptr_range]
- [`<[T]>::as_mut_ptr_range`][slice-as_mut_ptr_range]

Cargo
-----

No feature changes, but see compatibility notes.

Compatibility Notes
-------------------

- Previously native static libraries were linked as `whole-archive` in some cases, but now rustc tries not to use `whole-archive` unless explicitly requested. This [change][93901] may result in linking errors in some cases. To fix such errors, native libraries linked from the command line, build scripts, or [`#[link]` attributes][link-attr] need to
  - (more common) either be reordered to respect dependencies between them (if `a` depends on `b` then `a` should go first and `b` second)
  - (less common) or be updated to use the [`+whole-archive`] modifier.
- [Catching a second unwind from FFI code while cleaning up from a Rust panic now causes the process to abort][92911]
- [Proc macros no longer see `ident` matchers wrapped in groups][92472]
- [The number of `#` in `r#` raw string literals is now required to be less than 256][95251]
- [When checking that a dyn type satisfies a trait bound, supertrait bounds are now enforced][92285]
- [`cargo vendor` now only accepts one value for each `--sync` flag][cargo/10448]
- [`cfg` predicates in `all()` and `any()` are always evaluated to detect errors, instead of short-circuiting.][94295] The compatibility considerations here arise in nightly-only code that used the short-circuiting behavior of `all` to write something like `cfg(all(feature = "nightly", syntax-requiring-nightly))`, which will now fail to compile. Instead, use either `cfg_attr(feature = "nightly", ...)` or nested uses of `cfg`.
- [bootstrap: static-libstdcpp is now enabled by default, and can now be disabled when llvm-tools is enabled][94832]

Internal Changes
----------------

These changes provide no direct user facing benefits, but represent significant
improvements to the internals and overall performance of rustc
and related tools.

- [debuginfo: Refactor debuginfo generation for types][94261]
- [Remove the everybody loops pass][93913]

[88375]: https://github.com/rust-lang/rust/pull/88375/
[89887]: https://github.com/rust-lang/rust/pull/89887/
[90621]: https://github.com/rust-lang/rust/pull/90621/
[92285]: https://github.com/rust-lang/rust/pull/92285/
[92472]: https://github.com/rust-lang/rust/pull/92472/
[92697]: https://github.com/rust-lang/rust/pull/92697/
[92714]: https://github.com/rust-lang/rust/pull/92714/
[92911]: https://github.com/rust-lang/rust/pull/92911/
[93263]: https://github.com/rust-lang/rust/pull/93263/
[93745]: https://github.com/rust-lang/rust/pull/93745/
[93827]: https://github.com/rust-lang/rust/pull/93827/
[93901]: https://github.com/rust-lang/rust/pull/93901/
[93913]: https://github.com/rust-lang/rust/pull/93913/
[93965]: https://github.com/rust-lang/rust/pull/93965/
[94081]: https://github.com/rust-lang/rust/pull/94081/
[94261]: https://github.com/rust-lang/rust/pull/94261/
[94295]: https://github.com/rust-lang/rust/pull/94295/
[94832]: https://github.com/rust-lang/rust/pull/94832/
[95016]: https://github.com/rust-lang/rust/pull/95016/
[95251]: https://github.com/rust-lang/rust/pull/95251/
[`+whole-archive`]: https://doc.rust-lang.org/stable/rustc/command-line-arguments.html#linking-modifiers-whole-archive
[`Pin::static_mut`]: https://doc.rust-lang.org/stable/std/pin/struct.Pin.html#method.static_mut
[`Pin::static_ref`]: https://doc.rust-lang.org/stable/std/pin/struct.Pin.html#method.static_ref
[`Vec::retain_mut`]: https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#method.retain_mut
[`VecDeque::retain_mut`]: https://doc.rust-lang.org/stable/std/collections/struct.VecDeque.html#method.retain_mut
[`std::os::unix::net::SocketAddr::from_pathname`]: https://doc.rust-lang.org/stable/std/os/unix/net/struct.SocketAddr.html#method.from_pathname
[`std::process::ExitCode`]: https://doc.rust-lang.org/stable/std/process/struct.ExitCode.html
[`std::process::Termination`]: https://doc.rust-lang.org/stable/std/process/trait.Termination.html
[`std::thread::JoinHandle::is_finished`]: https://doc.rust-lang.org/stable/std/thread/struct.JoinHandle.html#method.is_finished
[cargo/10448]: https://github.com/rust-lang/cargo/pull/10448/
[cursor-write-array]: https://doc.rust-lang.org/stable/std/io/struct.Cursor.html#impl-Write-4
[link-attr]: https://doc.rust-lang.org/stable/reference/items/external-blocks.html#the-link-attribute
[ptr-add]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.add
[ptr-offset]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset
[ptr-sub]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.sub
[ptr-wrapping_add]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.wrapping_add
[ptr-wrapping_offset]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.wrapping_offset
[ptr-wrapping_sub]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.wrapping_sub
[slice-as_mut_ptr]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.as_mut_ptr
[slice-as_mut_ptr_range]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.as_mut_ptr_range
[slice-as_ptr_range]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.as_ptr_range
[target_feature]: https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute


Version 1.60.0 (2022-04-07)
==========================

Language
--------
- [Stabilize `#[cfg(panic = "...")]` for either `"unwind"` or `"abort"`.][93658]
- [Stabilize `#[cfg(target_has_atomic = "...")]` for each integer size and `"ptr"`.][93824]

Compiler
--------
- [Enable combining `+crt-static` and `relocation-model=pic` on `x86_64-unknown-linux-gnu`][86374]
- [Fixes wrong `unreachable_pub` lints on nested and glob public reexport][87487]
- [Stabilize `-Z instrument-coverage` as `-C instrument-coverage`][90132]
- [Stabilize `-Z print-link-args` as `--print link-args`][91606]
- [Add new Tier 3 target `mips64-openwrt-linux-musl`\*][92300]
- [Add new Tier 3 target `armv7-unknown-linux-uclibceabi` (softfloat)\*][92383]
- [Fix invalid removal of newlines from doc comments][92357]
- [Add kernel target for RustyHermit][92670]
- [Deny mixing bin crate type with lib crate types][92933]
- [Make rustc use `RUST_BACKTRACE=full` by default][93566]
- [Upgrade to LLVM 14][93577]

\* Refer to Rust's [platform support page][platform-support-doc] for more
   information on Rust's tiered platform support.

Libraries
---------
- [Guarantee call order for `sort_by_cached_key`][89621]
- [Improve `Duration::try_from_secs_f32`/`f64` accuracy by directly processing exponent and mantissa][90247]
- [Make `Instant::{duration_since, elapsed, sub}` saturating][89926]
- [Remove non-monotonic clocks workarounds in `Instant::now`][89926]
- [Make `BuildHasherDefault`, `iter::Empty` and `future::Pending` covariant][92630]

Stabilized APIs
---------------
- [`Arc::new_cyclic`][arc_new_cyclic]
- [`Rc::new_cyclic`][rc_new_cyclic]
- [`slice::EscapeAscii`][slice_escape_ascii]
- [`<[u8]>::escape_ascii`][slice_u8_escape_ascii]
- [`u8::escape_ascii`][u8_escape_ascii]
- [`Vec::spare_capacity_mut`][vec_spare_capacity_mut]
- [`MaybeUninit::assume_init_drop`][assume_init_drop]
- [`MaybeUninit::assume_init_read`][assume_init_read]
- [`i8::abs_diff`][i8_abs_diff]
- [`i16::abs_diff`][i16_abs_diff]
- [`i32::abs_diff`][i32_abs_diff]
- [`i64::abs_diff`][i64_abs_diff]
- [`i128::abs_diff`][i128_abs_diff]
- [`isize::abs_diff`][isize_abs_diff]
- [`u8::abs_diff`][u8_abs_diff]
- [`u16::abs_diff`][u16_abs_diff]
- [`u32::abs_diff`][u32_abs_diff]
- [`u64::abs_diff`][u64_abs_diff]
- [`u128::abs_diff`][u128_abs_diff]
- [`usize::abs_diff`][usize_abs_diff]
- [`Display for io::ErrorKind`][display_error_kind]
- [`From<u8> for ExitCode`][from_u8_exit_code]
- [`Not for !` (the "never" type)][not_never]
- [_Op_`Assign<$t> for Wrapping<$t>`][wrapping_assign_ops]
- [`arch::is_aarch64_feature_detected!`][is_aarch64_feature_detected]

Cargo
-----
- [Port cargo from `toml-rs` to `toml_edit`][cargo/10086]
- [Stabilize `-Ztimings` as `--timings`][cargo/10245]
- [Stabilize namespaced and weak dependency features.][cargo/10269]
- [Accept more `cargo:rustc-link-arg-*` types from build script output.][cargo/10274]
- [cargo-new should not add ignore rule on Cargo.lock inside subdirs][cargo/10379]

Misc
----
- [Ship docs on Tier 2 platforms by reusing the closest Tier 1 platform docs][92800]
- [Drop rustc-docs from complete profile][93742]
- [bootstrap: tidy up flag handling for llvm build][93918]

Compatibility Notes
-------------------
- [Remove compiler-rt linking hack on Android][83822]
- [Mitigations for platforms with non-monotonic clocks have been removed from
  `Instant::now`][89926]. On platforms that don't provide monotonic clocks, an
  instant is not guaranteed to be greater than an earlier instant anymore.
- [`Instant::{duration_since, elapsed, sub}` do not panic anymore on underflow,
  saturating to `0` instead][89926]. In the real world the panic happened mostly
  on platforms with buggy monotonic clock implementations rather than catching
  programming errors like reversing the start and end times. Such programming
  errors will now results in `0` rather than a panic.
- In a future release we're planning to increase the baseline requirements for
  the Linux kernel to version 3.2, and for glibc to version 2.17. We'd love
  your feedback in [PR #95026][95026].

Internal Changes
----------------

These changes provide no direct user facing benefits, but represent significant
improvements to the internals and overall performance of rustc
and related tools.

- [Switch all libraries to the 2021 edition][92068]

[83822]: https://github.com/rust-lang/rust/pull/83822
[86374]: https://github.com/rust-lang/rust/pull/86374
[87487]: https://github.com/rust-lang/rust/pull/87487
[89621]: https://github.com/rust-lang/rust/pull/89621
[89926]: https://github.com/rust-lang/rust/pull/89926
[90132]: https://github.com/rust-lang/rust/pull/90132
[90247]: https://github.com/rust-lang/rust/pull/90247
[91606]: https://github.com/rust-lang/rust/pull/91606
[92068]: https://github.com/rust-lang/rust/pull/92068
[92300]: https://github.com/rust-lang/rust/pull/92300
[92357]: https://github.com/rust-lang/rust/pull/92357
[92383]: https://github.com/rust-lang/rust/pull/92383
[92630]: https://github.com/rust-lang/rust/pull/92630
[92670]: https://github.com/rust-lang/rust/pull/92670
[92800]: https://github.com/rust-lang/rust/pull/92800
[92933]: https://github.com/rust-lang/rust/pull/92933
[93566]: https://github.com/rust-lang/rust/pull/93566
[93577]: https://github.com/rust-lang/rust/pull/93577
[93658]: https://github.com/rust-lang/rust/pull/93658
[93742]: https://github.com/rust-lang/rust/pull/93742
[93824]: https://github.com/rust-lang/rust/pull/93824
[93918]: https://github.com/rust-lang/rust/pull/93918
[95026]: https://github.com/rust-lang/rust/pull/95026

[cargo/10086]: https://github.com/rust-lang/cargo/pull/10086
[cargo/10245]: https://github.com/rust-lang/cargo/pull/10245
[cargo/10269]: https://github.com/rust-lang/cargo/pull/10269
[cargo/10274]: https://github.com/rust-lang/cargo/pull/10274
[cargo/10379]: https://github.com/rust-lang/cargo/pull/10379

[arc_new_cyclic]: https://doc.rust-lang.org/stable/std/sync/struct.Arc.html#method.new_cyclic
[rc_new_cyclic]: https://doc.rust-lang.org/stable/std/rc/struct.Rc.html#method.new_cyclic
[slice_escape_ascii]: https://doc.rust-lang.org/stable/std/slice/struct.EscapeAscii.html
[slice_u8_escape_ascii]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.escape_ascii
[u8_escape_ascii]: https://doc.rust-lang.org/stable/std/primitive.u8.html#method.escape_ascii
[vec_spare_capacity_mut]: https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#method.spare_capacity_mut
[assume_init_drop]: https://doc.rust-lang.org/stable/std/mem/union.MaybeUninit.html#method.assume_init_drop
[assume_init_read]: https://doc.rust-lang.org/stable/std/mem/union.MaybeUninit.html#method.assume_init_read
[i8_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.i8.html#method.abs_diff
[i16_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.i16.html#method.abs_diff
[i32_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.i32.html#method.abs_diff
[i64_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.i64.html#method.abs_diff
[i128_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.i128.html#method.abs_diff
[isize_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.isize.html#method.abs_diff
[u8_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.u8.html#method.abs_diff
[u16_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.u16.html#method.abs_diff
[u32_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.u32.html#method.abs_diff
[u64_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.u64.html#method.abs_diff
[u128_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.u128.html#method.abs_diff
[usize_abs_diff]: https://doc.rust-lang.org/stable/std/primitive.usize.html#method.abs_diff
[display_error_kind]: https://doc.rust-lang.org/stable/std/io/enum.ErrorKind.html#impl-Display
[from_u8_exit_code]: https://doc.rust-lang.org/stable/std/process/struct.ExitCode.html#impl-From%3Cu8%3E
[not_never]: https://doc.rust-lang.org/stable/std/primitive.never.html#impl-Not
[wrapping_assign_ops]: https://doc.rust-lang.org/stable/std/num/struct.Wrapping.html#trait-implementations
[is_aarch64_feature_detected]: https://doc.rust-lang.org/stable/std/arch/macro.is_aarch64_feature_detected.html

Version 1.59.0 (2022-02-24)
==========================

Language
--------

- [Stabilize default arguments for const parameters and remove the ordering restriction for type and const parameters][90207]
- [Stabilize destructuring assignment][90521]
- [Relax private in public lint on generic bounds and where clauses of trait impls][90586]
- [Stabilize asm! and global_asm! for x86, x86_64, ARM, Aarch64, and RISC-V][91728]

Compiler
--------

- [Stabilize new symbol mangling format, leaving it opt-in (-Csymbol-mangling-version=v0)][90128]
- [Emit LLVM optimization remarks when enabled with `-Cremark`][90833]
- [Fix sparc64 ABI for aggregates with floating point members][91003]
- [Warn when a `#[test]`-like built-in attribute macro is present multiple times.][91172]
- [Add support for riscv64gc-unknown-freebsd][91284]
- [Stabilize `-Z emit-future-incompat` as `--json future-incompat`][91535]
- [Soft disable incremental compilation][94124]

This release disables incremental compilation, unless the user has explicitly
opted in via the newly added RUSTC_FORCE_INCREMENTAL=1 environment variable.
This is due to a known and relatively frequently occurring bug in incremental
compilation, which causes builds to issue internal compiler errors. This
particular bug is already fixed on nightly, but that fix has not yet rolled out
to stable and is deemed too risky for a direct stable backport.

As always, we encourage users to test with nightly and report bugs so that we
can track failures and fix issues earlier.

See [94124] for more details.

[94124]: https://github.com/rust-lang/rust/issues/94124

Libraries
---------

- [Remove unnecessary bounds for some Hash{Map,Set} methods][91593]

Stabilized APIs
---------------

- [`std::thread::available_parallelism`][available_parallelism]
- [`Result::copied`][result-copied]
- [`Result::cloned`][result-cloned]
- [`arch::asm!`][asm]
- [`arch::global_asm!`][global_asm]
- [`ops::ControlFlow::is_break`][is_break]
- [`ops::ControlFlow::is_continue`][is_continue]
- [`TryFrom<char> for u8`][try_from_char_u8]
- [`char::TryFromCharError`][try_from_char_err]
  implementing `Clone`, `Debug`, `Display`, `PartialEq`, `Copy`, `Eq`, `Error`
- [`iter::zip`][zip]
- [`NonZeroU8::is_power_of_two`][is_power_of_two8]
- [`NonZeroU16::is_power_of_two`][is_power_of_two16]
- [`NonZeroU32::is_power_of_two`][is_power_of_two32]
- [`NonZeroU64::is_power_of_two`][is_power_of_two64]
- [`NonZeroU128::is_power_of_two`][is_power_of_two128]
- [`NonZeroUsize::is_power_of_two`][is_power_of_two_usize]
- [`DoubleEndedIterator for ToLowercase`][lowercase]
- [`DoubleEndedIterator for ToUppercase`][uppercase]
- [`TryFrom<&mut [T]> for [T; N]`][tryfrom_ref_arr]
- [`UnwindSafe for Once`][unwindsafe_once]
- [`RefUnwindSafe for Once`][refunwindsafe_once]
- [armv8 neon intrinsics for aarch64][stdarch/1266]

Const-stable:

- [`mem::MaybeUninit::as_ptr`][muninit_ptr]
- [`mem::MaybeUninit::assume_init`][muninit_init]
- [`mem::MaybeUninit::assume_init_ref`][muninit_init_ref]
- [`ffi::CStr::from_bytes_with_nul_unchecked`][cstr_from_bytes]

Cargo
-----

- [Stabilize the `strip` profile option][cargo/10088]
- [Stabilize future-incompat-report][cargo/10165]
- [Support abbreviating `--release` as `-r`][cargo/10133]
- [Support `term.quiet` configuration][cargo/10152]
- [Remove `--host` from cargo {publish,search,login}][cargo/10145]

Compatibility Notes
-------------------

- [Refactor weak symbols in std::sys::unix][90846]
  This may add new, versioned, symbols when building with a newer glibc, as the
  standard library uses weak linkage rather than dynamically attempting to load
  certain symbols at runtime.
- [Deprecate crate_type and crate_name nested inside `#![cfg_attr]`][83744]
  This adds a future compatibility lint to supporting the use of cfg_attr
  wrapping either crate_type or crate_name specification within Rust files;
  it is recommended that users migrate to setting the equivalent command line
  flags.
- [Remove effect of `#[no_link]` attribute on name resolution][92034]
  This may expose new names, leading to conflicts with preexisting names in a
  given namespace and a compilation failure.
- [Cargo will document libraries before binaries.][cargo/10172]
- [Respect doc=false in dependencies, not just the root crate][cargo/10201]
- [Weaken guarantee around advancing underlying iterators in zip][83791]
- [Make split_inclusive() on an empty slice yield an empty output][89825]
- [Update std::env::temp_dir to use GetTempPath2 on Windows when available.][89999]
- [unreachable! was updated to match other formatting macro behavior on Rust 2021][92137]

Internal Changes
----------------

These changes provide no direct user facing benefits, but represent significant
improvements to the internals and overall performance of rustc
and related tools.

- [Fix many cases of normalization-related ICEs][91255]
- [Replace dominators algorithm with simple Lengauer-Tarjan][85013]
- [Store liveness in interval sets for region inference][90637]

- [Remove `in_band_lifetimes` from the compiler and standard library, in preparation for removing this
  unstable feature.][91867]

[91867]: https://github.com/rust-lang/rust/issues/91867
[83744]: https://github.com/rust-lang/rust/pull/83744/
[83791]: https://github.com/rust-lang/rust/pull/83791/
[85013]: https://github.com/rust-lang/rust/pull/85013/
[89825]: https://github.com/rust-lang/rust/pull/89825/
[89999]: https://github.com/rust-lang/rust/pull/89999/
[90128]: https://github.com/rust-lang/rust/pull/90128/
[90207]: https://github.com/rust-lang/rust/pull/90207/
[90521]: https://github.com/rust-lang/rust/pull/90521/
[90586]: https://github.com/rust-lang/rust/pull/90586/
[90637]: https://github.com/rust-lang/rust/pull/90637/
[90833]: https://github.com/rust-lang/rust/pull/90833/
[90846]: https://github.com/rust-lang/rust/pull/90846/
[91003]: https://github.com/rust-lang/rust/pull/91003/
[91172]: https://github.com/rust-lang/rust/pull/91172/
[91255]: https://github.com/rust-lang/rust/pull/91255/
[91284]: https://github.com/rust-lang/rust/pull/91284/
[91535]: https://github.com/rust-lang/rust/pull/91535/
[91593]: https://github.com/rust-lang/rust/pull/91593/
[91728]: https://github.com/rust-lang/rust/pull/91728/
[91878]: https://github.com/rust-lang/rust/pull/91878/
[91896]: https://github.com/rust-lang/rust/pull/91896/
[91926]: https://github.com/rust-lang/rust/pull/91926/
[91984]: https://github.com/rust-lang/rust/pull/91984/
[92020]: https://github.com/rust-lang/rust/pull/92020/
[92034]: https://github.com/rust-lang/rust/pull/92034/
[92137]: https://github.com/rust-lang/rust/pull/92137/
[92483]: https://github.com/rust-lang/rust/pull/92483/
[cargo/10088]: https://github.com/rust-lang/cargo/pull/10088/
[cargo/10133]: https://github.com/rust-lang/cargo/pull/10133/
[cargo/10145]: https://github.com/rust-lang/cargo/pull/10145/
[cargo/10152]: https://github.com/rust-lang/cargo/pull/10152/
[cargo/10165]: https://github.com/rust-lang/cargo/pull/10165/
[cargo/10172]: https://github.com/rust-lang/cargo/pull/10172/
[cargo/10201]: https://github.com/rust-lang/cargo/pull/10201/
[cargo/10269]: https://github.com/rust-lang/cargo/pull/10269/

[cstr_from_bytes]: https://doc.rust-lang.org/stable/std/ffi/struct.CStr.html#method.from_bytes_with_nul_unchecked
[muninit_ptr]: https://doc.rust-lang.org/stable/std/mem/union.MaybeUninit.html#method.as_ptr
[muninit_init]: https://doc.rust-lang.org/stable/std/mem/union.MaybeUninit.html#method.assume_init
[muninit_init_ref]: https://doc.rust-lang.org/stable/std/mem/union.MaybeUninit.html#method.assume_init_ref
[unwindsafe_once]: https://doc.rust-lang.org/stable/std/sync/struct.Once.html#impl-UnwindSafe
[refunwindsafe_once]: https://doc.rust-lang.org/stable/std/sync/struct.Once.html#impl-RefUnwindSafe
[tryfrom_ref_arr]: https://doc.rust-lang.org/stable/std/convert/trait.TryFrom.html#impl-TryFrom%3C%26%27_%20mut%20%5BT%5D%3E
[lowercase]: https://doc.rust-lang.org/stable/std/char/struct.ToLowercase.html#impl-DoubleEndedIterator
[uppercase]: https://doc.rust-lang.org/stable/std/char/struct.ToUppercase.html#impl-DoubleEndedIterator
[try_from_char_err]: https://doc.rust-lang.org/stable/std/char/struct.TryFromCharError.html
[available_parallelism]: https://doc.rust-lang.org/stable/std/thread/fn.available_parallelism.html
[result-copied]: https://doc.rust-lang.org/stable/std/result/enum.Result.html#method.copied
[result-cloned]: https://doc.rust-lang.org/stable/std/result/enum.Result.html#method.cloned
[asm]: https://doc.rust-lang.org/stable/core/arch/macro.asm.html
[global_asm]: https://doc.rust-lang.org/stable/core/arch/macro.global_asm.html
[is_break]: https://doc.rust-lang.org/stable/std/ops/enum.ControlFlow.html#method.is_break
[is_continue]: https://doc.rust-lang.org/stable/std/ops/enum.ControlFlow.html#method.is_continue
[try_from_char_u8]: https://doc.rust-lang.org/stable/std/primitive.char.html#impl-TryFrom%3Cchar%3E
[zip]: https://doc.rust-lang.org/stable/std/iter/fn.zip.html
[is_power_of_two8]: https://doc.rust-lang.org/stable/core/num/struct.NonZeroU8.html#method.is_power_of_two
[is_power_of_two16]: https://doc.rust-lang.org/stable/core/num/struct.NonZeroU16.html#method.is_power_of_two
[is_power_of_two32]: https://doc.rust-lang.org/stable/core/num/struct.NonZeroU32.html#method.is_power_of_two
[is_power_of_two64]: https://doc.rust-lang.org/stable/core/num/struct.NonZeroU64.html#method.is_power_of_two
[is_power_of_two128]: https://doc.rust-lang.org/stable/core/num/struct.NonZeroU128.html#method.is_power_of_two
[is_power_of_two_usize]: https://doc.rust-lang.org/stable/core/num/struct.NonZeroUsize.html#method.is_power_of_two
[stdarch/1266]: https://github.com/rust-lang/stdarch/pull/1266

Version 1.58.1 (2022-01-20)
===========================

* Fix race condition in `std::fs::remove_dir_all` ([CVE-2022-21658])
* [Handle captured arguments in the `useless_format` Clippy lint][clippy/8295]
* [Move `non_send_fields_in_send_ty` Clippy lint to nursery][clippy/8075]
* [Fix wrong error message displayed when some imports are missing][91254]
* [Fix rustfmt not formatting generated files from stdin][92912]

[CVE-2022-21658]: https://www.cve.org/CVERecord?id=CVE-2022-21658
[91254]: https://github.com/rust-lang/rust/pull/91254
[92912]: https://github.com/rust-lang/rust/pull/92912
[clippy/8075]: https://github.com/rust-lang/rust-clippy/pull/8075
[clippy/8295]: https://github.com/rust-lang/rust-clippy/pull/8295

Version 1.58.0 (2022-01-13)
==========================

Language
--------

- [Format strings can now capture arguments simply by writing `{ident}` in the string.][90473] This works in all macros accepting format strings. Support for this in `panic!` (`panic!("{ident}")`) requires the 2021 edition; panic invocations in previous editions that appear to be trying to use this will result in a warning lint about not having the intended effect.
- [`*const T` pointers can now be dereferenced in const contexts.][89551]
- [The rules for when a generic struct implements `Unsize` have been relaxed.][90417]

Compiler
--------

- [Add LLVM CFI support to the Rust compiler][89652]
- [Stabilize -Z strip as -C strip][90058]. Note that while release builds already don't add debug symbols for the code you compile, the compiled standard library that ships with Rust includes debug symbols, so you may want to use the `strip` option to remove these symbols to produce smaller release binaries. Note that this release only includes support in rustc, not directly in cargo.
- [Add support for LLVM coverage mapping format versions 5 and 6][91207]
- [Emit LLVM optimization remarks when enabled with `-Cremark`][90833]
- [Update the minimum external LLVM to 12][90175]
- [Add `x86_64-unknown-none` at Tier 3*][89062]
- [Build musl dist artifacts with debuginfo enabled][90733]. When building release binaries using musl, you may want to use the newly stabilized strip option to remove these debug symbols, reducing the size of your binaries.
- [Don't abort compilation after giving a lint error][87337]
- [Error messages point at the source of trait bound obligations in more places][89580]

\* Refer to Rust's [platform support page][platform-support-doc] for more
   information on Rust's tiered platform support.

Libraries
---------

- [All remaining functions in the standard library have `#[must_use]` annotations where appropriate][89692], producing a warning when ignoring their return value. This helps catch mistakes such as expecting a function to mutate a value in place rather than return a new value.
- [Paths are automatically canonicalized on Windows for operations that support it][89174]
- [Re-enable debug checks for `copy` and `copy_nonoverlapping`][90041]
- [Implement `RefUnwindSafe` for `Rc<T>`][87467]
- [Make RSplit<T, P>: Clone not require T: Clone][90117]
- [Implement `Termination` for `Result<Infallible, E>`][88601]. This allows writing `fn main() -> Result<Infallible, ErrorType>`, for a program whose successful exits never involve returning from `main` (for instance, a program that calls `exit`, or that uses `exec` to run another program).

Stabilized APIs
---------------

- [`Metadata::is_symlink`]
- [`Path::is_symlink`]
- [`{integer}::saturating_div`]
- [`Option::unwrap_unchecked`]
- [`Result::unwrap_unchecked`]
- [`Result::unwrap_err_unchecked`]
- [`File::options`]

These APIs are now usable in const contexts:

- [`Duration::new`]
- [`Duration::checked_add`]
- [`Duration::saturating_add`]
- [`Duration::checked_sub`]
- [`Duration::saturating_sub`]
- [`Duration::checked_mul`]
- [`Duration::saturating_mul`]
- [`Duration::checked_div`]

Cargo
-----

- [Add --message-format for install command][cargo/10107]
- [Warn when alias shadows external subcommand][cargo/10082]

Rustdoc
-------

- [Show all Deref implementations recursively in rustdoc][90183]
- [Use computed visibility in rustdoc][88447]

Compatibility Notes
-------------------

- [Try all stable method candidates first before trying unstable ones][90329]. This change ensures that adding new nightly-only methods to the Rust standard library will not break code invoking methods of the same name from traits outside the standard library.
- Windows: [`std::process::Command` will no longer search the current directory for executables.][87704]
- [All proc-macro backward-compatibility lints are now deny-by-default.][88041]
- [proc_macro: Append .0 to unsuffixed float if it would otherwise become int token][90297]
- [Refactor weak symbols in std::sys::unix][90846]. This optimizes accesses to glibc functions, by avoiding the use of dlopen. This does not increase the [minimum expected version of glibc](https://doc.rust-lang.org/nightly/rustc/platform-support.html). However, software distributions that use symbol versions to detect library dependencies, and which take weak symbols into account in that analysis, may detect rust binaries as requiring newer versions of glibc.
- [rustdoc now rejects some unexpected semicolons in doctests][91026]

Internal Changes
----------------

These changes provide no direct user facing benefits, but represent significant
improvements to the internals and overall performance of rustc
and related tools.

- [Implement coherence checks for negative trait impls][90104]
- [Add rustc lint, warning when iterating over hashmaps][89558]
- [Optimize live point computation][90491]
- [Enable verification for 1/32nd of queries loaded from disk][90361]
- [Implement version of normalize_erasing_regions that allows for normalization failure][91255]

[87337]: https://github.com/rust-lang/rust/pull/87337/
[87467]: https://github.com/rust-lang/rust/pull/87467/
[87704]: https://github.com/rust-lang/rust/pull/87704/
[88041]: https://github.com/rust-lang/rust/pull/88041/
[88447]: https://github.com/rust-lang/rust/pull/88447/
[88601]: https://github.com/rust-lang/rust/pull/88601/
[89062]: https://github.com/rust-lang/rust/pull/89062/
[89174]: https://github.com/rust-lang/rust/pull/89174/
[89551]: https://github.com/rust-lang/rust/pull/89551/
[89558]: https://github.com/rust-lang/rust/pull/89558/
[89580]: https://github.com/rust-lang/rust/pull/89580/
[89652]: https://github.com/rust-lang/rust/pull/89652/
[90041]: https://github.com/rust-lang/rust/pull/90041/
[90058]: https://github.com/rust-lang/rust/pull/90058/
[90104]: https://github.com/rust-lang/rust/pull/90104/
[90117]: https://github.com/rust-lang/rust/pull/90117/
[90175]: https://github.com/rust-lang/rust/pull/90175/
[90183]: https://github.com/rust-lang/rust/pull/90183/
[90297]: https://github.com/rust-lang/rust/pull/90297/
[90329]: https://github.com/rust-lang/rust/pull/90329/
[90361]: https://github.com/rust-lang/rust/pull/90361/
[90417]: https://github.com/rust-lang/rust/pull/90417/
[90473]: https://github.com/rust-lang/rust/pull/90473/
[90491]: https://github.com/rust-lang/rust/pull/90491/
[90733]: https://github.com/rust-lang/rust/pull/90733/
[90833]: https://github.com/rust-lang/rust/pull/90833/
[90846]: https://github.com/rust-lang/rust/pull/90846/
[91026]: https://github.com/rust-lang/rust/pull/91026/
[91207]: https://github.com/rust-lang/rust/pull/91207/
[91255]: https://github.com/rust-lang/rust/pull/91255/
[cargo/10082]: https://github.com/rust-lang/cargo/pull/10082/
[cargo/10107]: https://github.com/rust-lang/cargo/pull/10107/
[`Metadata::is_symlink`]: https://doc.rust-lang.org/stable/std/fs/struct.Metadata.html#method.is_symlink
[`Path::is_symlink`]: https://doc.rust-lang.org/stable/std/path/struct.Path.html#method.is_symlink
[`{integer}::saturating_div`]: https://doc.rust-lang.org/stable/std/primitive.i8.html#method.saturating_div
[`Option::unwrap_unchecked`]: https://doc.rust-lang.org/stable/std/option/enum.Option.html#method.unwrap_unchecked
[`Result::unwrap_unchecked`]: https://doc.rust-lang.org/stable/std/result/enum.Result.html#method.unwrap_unchecked
[`Result::unwrap_err_unchecked`]: https://doc.rust-lang.org/stable/std/result/enum.Result.html#method.unwrap_err_unchecked
[`File::options`]: https://doc.rust-lang.org/stable/std/fs/struct.File.html#method.options
[`Duration::new`]: https://doc.rust-lang.org/stable/std/time/struct.Duration.html#method.new

Version 1.57.0 (2021-12-02)
==========================

Language
--------

- [Macro attributes may follow `#[derive]` and will see the original (pre-`cfg`) input.][87220]
- [Accept curly-brace macros in expressions, like `m!{ .. }.method()` and `m!{ .. }?`.][88690]
- [Allow panicking in constant evaluation.][89508]
- [Ignore derived `Clone` and `Debug` implementations during dead code analysis.][85200]

Compiler
--------

- [Create more accurate debuginfo for vtables.][89597]
- [Add `armv6k-nintendo-3ds` at Tier 3\*.][88529]
- [Add `armv7-unknown-linux-uclibceabihf` at Tier 3\*.][88952]
- [Add `m68k-unknown-linux-gnu` at Tier 3\*.][88321]
- [Add SOLID targets at Tier 3\*:][86191] `aarch64-kmc-solid_asp3`, `armv7a-kmc-solid_asp3-eabi`, `armv7a-kmc-solid_asp3-eabihf`

\* Refer to Rust's [platform support page][platform-support-doc] for more
   information on Rust's tiered platform support.

Libraries
---------

- [Avoid allocations and copying in `Vec::leak`][89337]
- [Add `#[repr(i8)]` to `Ordering`][89507]
- [Optimize `File::read_to_end` and `read_to_string`][89582]
- [Update to Unicode 14.0][89614]
- [Many more functions are marked `#[must_use]`][89692], producing a warning
  when ignoring their return value. This helps catch mistakes such as expecting
  a function to mutate a value in place rather than return a new value.

Stabilised APIs
---------------

- [`[T; N]::as_mut_slice`][`array::as_mut_slice`]
- [`[T; N]::as_slice`][`array::as_slice`]
- [`collections::TryReserveError`]
- [`HashMap::try_reserve`]
- [`HashSet::try_reserve`]
- [`String::try_reserve`]
- [`String::try_reserve_exact`]
- [`Vec::try_reserve`]
- [`Vec::try_reserve_exact`]
- [`VecDeque::try_reserve`]
- [`VecDeque::try_reserve_exact`]
- [`Iterator::map_while`]
- [`iter::MapWhile`]
- [`proc_macro::is_available`]
- [`Command::get_program`]
- [`Command::get_args`]
- [`Command::get_envs`]
- [`Command::get_current_dir`]
- [`CommandArgs`]
- [`CommandEnvs`]

These APIs are now usable in const contexts:

- [`hint::unreachable_unchecked`]

Cargo
-----

- [Stabilize custom profiles][cargo/9943]

Compatibility notes
-------------------

- [Ignore derived `Clone` and `Debug` implementations during dead code analysis.][85200]
  This will break some builds that set `#![deny(dead_code)]`.

Internal changes
----------------
These changes provide no direct user facing benefits, but represent significant
improvements to the internals and overall performance of rustc
and related tools.

- [Added an experimental backend for codegen with `libgccjit`.][87260]

[85200]: https://github.com/rust-lang/rust/pull/85200/
[86191]: https://github.com/rust-lang/rust/pull/86191/
[87220]: https://github.com/rust-lang/rust/pull/87220/
[87260]: https://github.com/rust-lang/rust/pull/87260/
[88321]: https://github.com/rust-lang/rust/pull/88321/
[88529]: https://github.com/rust-lang/rust/pull/88529/
[88690]: https://github.com/rust-lang/rust/pull/88690/
[88952]: https://github.com/rust-lang/rust/pull/88952/
[89337]: https://github.com/rust-lang/rust/pull/89337/
[89507]: https://github.com/rust-lang/rust/pull/89507/
[89508]: https://github.com/rust-lang/rust/pull/89508/
[89582]: https://github.com/rust-lang/rust/pull/89582/
[89597]: https://github.com/rust-lang/rust/pull/89597/
[89614]: https://github.com/rust-lang/rust/pull/89614/
[89692]: https://github.com/rust-lang/rust/issues/89692/
[cargo/9943]: https://github.com/rust-lang/cargo/pull/9943/
[`array::as_mut_slice`]: https://doc.rust-lang.org/std/primitive.array.html#method.as_mut_slice
[`array::as_slice`]: https://doc.rust-lang.org/std/primitive.array.html#method.as_slice
[`collections::TryReserveError`]: https://doc.rust-lang.org/std/collections/struct.TryReserveError.html
[`HashMap::try_reserve`]: https://doc.rust-lang.org/std/collections/hash_map/struct.HashMap.html#method.try_reserve
[`HashSet::try_reserve`]: https://doc.rust-lang.org/std/collections/hash_set/struct.HashSet.html#method.try_reserve
[`String::try_reserve`]: https://doc.rust-lang.org/alloc/string/struct.String.html#method.try_reserve
[`String::try_reserve_exact`]: https://doc.rust-lang.org/alloc/string/struct.String.html#method.try_reserve_exact
[`Vec::try_reserve`]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.try_reserve
[`Vec::try_reserve_exact`]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.try_reserve_exact
[`VecDeque::try_reserve`]: https://doc.rust-lang.org/std/collections/struct.VecDeque.html#method.try_reserve
[`VecDeque::try_reserve_exact`]: https://doc.rust-lang.org/std/collections/struct.VecDeque.html#method.try_reserve_exact
[`Iterator::map_while`]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.map_while
[`iter::MapWhile`]: https://doc.rust-lang.org/std/iter/struct.MapWhile.html
[`proc_macro::is_available`]: https://doc.rust-lang.org/proc_macro/fn.is_available.html
[`Command::get_program`]: https://doc.rust-lang.org/std/process/struct.Command.html#method.get_program
[`Command::get_args`]: https://doc.rust-lang.org/std/process/struct.Command.html#method.get_args
[`Command::get_envs`]: https://doc.rust-lang.org/std/process/struct.Command.html#method.get_envs
[`Command::get_current_dir`]: https://doc.rust-lang.org/std/process/struct.Command.html#method.get_current_dir
[`CommandArgs`]: https://doc.rust-lang.org/std/process/struct.CommandArgs.html
[`CommandEnvs`]: https://doc.rust-lang.org/std/process/struct.CommandEnvs.html

Version 1.56.1 (2021-11-01)
===========================

- New lints to detect the presence of bidirectional-override Unicode
  codepoints in the compiled source code ([CVE-2021-42574])

[CVE-2021-42574]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-42574

Version 1.56.0 (2021-10-21)
========================

Language
--------

- [The 2021 Edition is now stable.][rust#88100]
  See [the edition guide][rust-2021-edition-guide] for more details.
- [The pattern in `binding @ pattern` can now also introduce new bindings.][rust#85305]
- [Union field access is permitted in `const fn`.][rust#85769]

[rust-2021-edition-guide]: https://doc.rust-lang.org/nightly/edition-guide/rust-2021/index.html

Compiler
--------

- [Upgrade to LLVM 13.][rust#87570]
- [Support memory, address, and thread sanitizers on aarch64-unknown-freebsd.][rust#88023]
- [Allow specifying a deployment target version for all iOS targets][rust#87699]
- [Warnings can be forced on with `--force-warn`.][rust#87472]
  This feature is primarily intended for usage by `cargo fix`, rather than end users.
- [Promote `aarch64-apple-ios-sim` to Tier 2\*.][rust#87760]
- [Add `powerpc-unknown-freebsd` at Tier 3\*.][rust#87370]
- [Add `riscv32imc-esp-espidf` at Tier 3\*.][rust#87666]

\* Refer to Rust's [platform support page][platform-support-doc] for more
information on Rust's tiered platform support.

Libraries
---------

- [Allow writing of incomplete UTF-8 sequences via stdout/stderr on Windows.][rust#83342]
  The Windows console still requires valid Unicode, but this change allows
  splitting a UTF-8 character across multiple write calls. This allows, for
  instance, programs that just read and write data buffers (e.g. copying a file
  to stdout) without regard for Unicode or character boundaries.
- [Prefer `AtomicU{64,128}` over Mutex for Instant backsliding protection.][rust#83093]
  For this use case, atomics scale much better under contention.
- [Implement `Extend<(A, B)>` for `(Extend<A>, Extend<B>)`][rust#85835]
- [impl Default, Copy, Clone for std::io::Sink and std::io::Empty][rust#86744]
- [`impl From<[(K, V); N]>` for all collections.][rust#84111]
- [Remove `P: Unpin` bound on impl Future for Pin.][rust#81363]
- [Treat invalid environment variable names as nonexistent.][rust#86183]
  Previously, the environment functions would panic if given a variable name
  with an internal null character or equal sign (`=`). Now, these functions will
  just treat such names as nonexistent variables, since the OS cannot represent
  the existence of a variable with such a name.

Stabilised APIs
---------------

- [`std::os::unix::fs::chroot`]
- [`UnsafeCell::raw_get`]
- [`BufWriter::into_parts`]
- [`core::panic::{UnwindSafe, RefUnwindSafe, AssertUnwindSafe}`]
  These APIs were previously stable in `std`, but are now also available in `core`.
- [`Vec::shrink_to`]
- [`String::shrink_to`]
- [`OsString::shrink_to`]
- [`PathBuf::shrink_to`]
- [`BinaryHeap::shrink_to`]
- [`VecDeque::shrink_to`]
- [`HashMap::shrink_to`]
- [`HashSet::shrink_to`]

These APIs are now usable in const contexts:

- [`std::mem::transmute`]
- [`[T]::first`][`slice::first`]
- [`[T]::split_first`][`slice::split_first`]
- [`[T]::last`][`slice::last`]
- [`[T]::split_last`][`slice::split_last`]

Cargo
-----

- [Cargo supports specifying a minimum supported Rust version in Cargo.toml.][`rust-version`]
  This has no effect at present on dependency version selection.
  We encourage crates to specify their minimum supported Rust version, and we encourage CI systems
  that support Rust code to include a crate's specified minimum version in the test matrix for that
  crate by default.

Compatibility notes
-------------------

- [Update to new argument parsing rules on Windows.][rust#87580]
  This adjusts Rust's standard library to match the behavior of the standard
  libraries for C/C++. The rules have changed slightly over time, and this PR
  brings us to the latest set of rules (changed in 2008).
- [Disallow the aapcs calling convention on aarch64][rust#88399]
  This was already not supported by LLVM; this change surfaces this lack of
  support with a better error message.
- [Make `SEMICOLON_IN_EXPRESSIONS_FROM_MACROS` warn by default][rust#87385]
- [Warn when an escaped newline skips multiple lines.][rust#87671]
- [Calls to `libc::getpid` / `std::process::id` from `Command::pre_exec`
   may return different values on glibc <= 2.24.][rust#81825]
   Rust now invokes the `clone3` system call directly, when available, to use new functionality
   available via that system call. Older versions of glibc cache the result of `getpid`, and only
   update that cache when calling glibc's clone/fork functions, so a direct system call bypasses
   that cache update. glibc 2.25 and newer no longer cache `getpid` for exactly this reason.

Internal changes
----------------
These changes provide no direct user facing benefits, but represent significant
improvements to the internals and overall performance of rustc
and related tools.

- [LLVM is compiled with PGO in published x86_64-unknown-linux-gnu artifacts.][rust#88069]
  This improves the performance of most Rust builds.
- [Unify representation of macros in internal data structures.][rust#88019]
  This change fixes a host of bugs with the handling of macros by the compiler,
  as well as rustdoc.

[`std::os::unix::fs::chroot`]: https://doc.rust-lang.org/stable/std/os/unix/fs/fn.chroot.html
[`UnsafeCell::raw_get`]: https://doc.rust-lang.org/stable/std/cell/struct.UnsafeCell.html#method.raw_get
[`BufWriter::into_parts`]: https://doc.rust-lang.org/stable/std/io/struct.BufWriter.html#method.into_parts
[`core::panic::{UnwindSafe, RefUnwindSafe, AssertUnwindSafe}`]: https://github.com/rust-lang/rust/pull/84662
[`Vec::shrink_to`]: https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#method.shrink_to
[`String::shrink_to`]: https://doc.rust-lang.org/stable/std/string/struct.String.html#method.shrink_to
[`OsString::shrink_to`]: https://doc.rust-lang.org/stable/std/ffi/struct.OsString.html#method.shrink_to
[`PathBuf::shrink_to`]: https://doc.rust-lang.org/stable/std/path/struct.PathBuf.html#method.shrink_to
[`BinaryHeap::shrink_to`]: https://doc.rust-lang.org/stable/std/collections/struct.BinaryHeap.html#method.shrink_to
[`VecDeque::shrink_to`]: https://doc.rust-lang.org/stable/std/collections/struct.VecDeque.html#method.shrink_to
[`HashMap::shrink_to`]: https://doc.rust-lang.org/stable/std/collections/hash_map/struct.HashMap.html#method.shrink_to
[`HashSet::shrink_to`]: https://doc.rust-lang.org/stable/std/collections/hash_set/struct.HashSet.html#method.shrink_to
[`std::mem::transmute`]: https://doc.rust-lang.org/stable/std/mem/fn.transmute.html
[`slice::first`]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.first
[`slice::split_first`]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.split_first
[`slice::last`]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.last
[`slice::split_last`]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.split_last
[`rust-version`]: https://doc.rust-lang.org/nightly/cargo/reference/manifest.html#the-rust-version-field
[rust#87671]: https://github.com/rust-lang/rust/pull/87671
[rust#86183]: https://github.com/rust-lang/rust/pull/86183
[rust#87385]: https://github.com/rust-lang/rust/pull/87385
[rust#88100]: https://github.com/rust-lang/rust/pull/88100
[rust#85305]: https://github.com/rust-lang/rust/pull/85305
[rust#88069]: https://github.com/rust-lang/rust/pull/88069
[rust#87472]: https://github.com/rust-lang/rust/pull/87472
[rust#87699]: https://github.com/rust-lang/rust/pull/87699
[rust#87570]: https://github.com/rust-lang/rust/pull/87570
[rust#88023]: https://github.com/rust-lang/rust/pull/88023
[rust#87760]: https://github.com/rust-lang/rust/pull/87760
[rust#87370]: https://github.com/rust-lang/rust/pull/87370
[rust#87580]: https://github.com/rust-lang/rust/pull/87580
[rust#83342]: https://github.com/rust-lang/rust/pull/83342
[rust#83093]: https://github.com/rust-lang/rust/pull/83093
[rust#85835]: https://github.com/rust-lang/rust/pull/85835
[rust#86744]: https://github.com/rust-lang/rust/pull/86744
[rust#81363]: https://github.com/rust-lang/rust/pull/81363
[rust#84111]: https://github.com/rust-lang/rust/pull/84111
[rust#85769]: https://github.com/rust-lang/rust/pull/85769#issuecomment-854363720
[rust#88399]: https://github.com/rust-lang/rust/pull/88399
[rust#81825]: https://github.com/rust-lang/rust/pull/81825#issuecomment-808406918
[rust#88019]: https://github.com/rust-lang/rust/pull/88019
[rust#87666]: https://github.com/rust-lang/rust/pull/87666
[platform-support-doc]: https://doc.rust-lang.org/nightly/rustc/platform-support.html
[`Duration::checked_add`]: https://doc.rust-lang.org/std/time/struct.Duration.html#method.checked_add
[`Duration::checked_div`]: https://doc.rust-lang.org/std/time/struct.Duration.html#method.checked_div
[`Duration::checked_mul`]: https://doc.rust-lang.org/std/time/struct.Duration.html#method.checked_mul
[`Duration::checked_sub`]: https://doc.rust-lang.org/std/time/struct.Duration.html#method.checked_sub
[`Duration::saturating_add`]: https://doc.rust-lang.org/std/time/struct.Duration.html#method.saturating_add
[`Duration::saturating_mul`]: https://doc.rust-lang.org/std/time/struct.Duration.html#method.saturating_mul
[`Duration::saturating_sub`]: https://doc.rust-lang.org/std/time/struct.Duration.html#method.saturating_sub
[`hint::unreachable_unchecked`]: https://doc.rust-lang.org/std/hint/fn.unreachable_unchecked.html
