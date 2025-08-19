# Libraries and metadata

When the compiler sees a reference to an external crate, it needs to load some
information about that crate. This chapter gives an overview of that process,
and the supported file formats for crate libraries.

## Libraries

A crate dependency can be loaded from an `rlib`, `dylib`, or `rmeta` file. A
key point of these file formats is that they contain `rustc`-specific
[*metadata*](#metadata). This metadata allows the compiler to discover enough
information about the external crate to understand the items it contains,
which macros it exports, and *much* more.

### rlib

An `rlib` is an [archive file], which is similar to a tar file. This file
format is specific to `rustc`, and may change over time. This file contains:

* Object code, which is the result of code generation. This is used during
  regular linking. There is a separate `.o` file for each [codegen unit]. The
  codegen step can be skipped with the [`-C
  linker-plugin-lto`][linker-plugin-lto] CLI option, which means each `.o`
  file will only contain LLVM bitcode.
* [LLVM bitcode], which is a binary representation of LLVM's intermediate
  representation, which is embedded as a section in the `.o` files. This can
  be used for [Link Time Optimization] (LTO). This can be removed with the
  [`-C embed-bitcode=no`][embed-bitcode] CLI option to improve compile times
  and reduce disk space if LTO is not needed.
* `rustc` [metadata], in a file named `lib.rmeta`.
* A symbol table, which is essentially a list of symbols with offsets to the
  object files that contain that symbol. This is pretty standard for archive
  files.

[archive file]: https://en.wikipedia.org/wiki/Ar_(Unix)
[LLVM bitcode]: https://llvm.org/docs/BitCodeFormat.html
[Link Time Optimization]: https://llvm.org/docs/LinkTimeOptimization.html
[codegen unit]: ../backend/codegen.md
[embed-bitcode]: https://doc.rust-lang.org/rustc/codegen-options/index.html#embed-bitcode
[linker-plugin-lto]: https://doc.rust-lang.org/rustc/codegen-options/index.html#linker-plugin-lto

### dylib

A `dylib` is a platform-specific shared library. It includes the `rustc`
[metadata] in a special link section called `.rustc`.

### rmeta

An `rmeta` file is a custom binary format that contains the [metadata] for the
crate. This file can be used for fast "checks" of a project by skipping all code
generation (as is done with `cargo check`), collecting enough information for
documentation (as is done with `cargo doc`), or for [pipelining](#pipelining).
This file is created if the [`--emit=metadata`][emit] CLI option is used.

`rmeta` files do not support linking, since they do not contain compiled
object files.

[emit]: https://doc.rust-lang.org/rustc/command-line-arguments.html#option-emit

## Metadata

The metadata contains a wide swath of different elements. This guide will not go
into detail about every field it contains. You are encouraged to browse the
[`CrateRoot`] definition to get a sense of the different elements it contains.
Everything about metadata encoding and decoding is in the [`rustc_metadata`]
package.

Here are a few highlights of things it contains:

* The version of the `rustc` compiler. The compiler will refuse to load files
  from any other version.
* The [Strict Version Hash](#strict-version-hash) (SVH). This helps ensure the
  correct dependency is loaded.
* The [Stable Crate Id](#stable-crate-id). This is a hash used
  to identify crates.
* Information about all the source files in the library. This can be used for
  a variety of things, such as diagnostics pointing to sources in a
  dependency.
* Information about exported macros, traits, types, and items. Generally,
  anything that's needed to be known when a path references something inside a
  crate dependency.
* Encoded [MIR]. This is optional, and only encoded if needed for code
  generation. `cargo check` skips this for performance reasons.

[`CrateRoot`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/rmeta/struct.CrateRoot.html
[`rustc_metadata`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/index.html
[MIR]: ../mir/index.md

### Strict Version Hash

The Strict Version Hash ([SVH], also known as the "crate hash") is a 64-bit
hash that is used to ensure that the correct crate dependencies are loaded. It
is possible for a directory to contain multiple copies of the same dependency
built with different settings, or built from different sources. The crate
loader will skip any crates that have the wrong SVH.

The SVH is also used for the [incremental compilation] session filename,
though that usage is mostly historic.

The hash includes a variety of elements:

* Hashes of the HIR nodes.
* All of the upstream crate hashes.
* All of the source filenames.
* Hashes of certain command-line flags (like `-C metadata` via the [Stable
  Crate Id](#stable-crate-id), and all CLI options marked with `[TRACKED]`).

See [`compute_hir_hash`] for where the hash is actually computed.

[SVH]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_data_structures/svh/struct.Svh.html
[incremental compilation]: ../queries/incremental-compilation.md
[`compute_hir_hash`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast_lowering/fn.compute_hir_hash.html

### Stable Crate Id

The [`StableCrateId`] is a 64-bit hash used to identify different crates with
potentially the same name. It is a hash of the crate name and all the
[`-C metadata`] CLI options computed in [`StableCrateId::new`]. It is
used in a variety of places, such as symbol name mangling, crate loading, and
much more.

By default, all Rust symbols are mangled and incorporate the stable crate id.
This allows multiple versions of the same crate to be included together. Cargo
automatically generates `-C metadata` hashes based on a variety of factors, like
the package version, source, and target kind (a lib and test can have the same
crate name, so they need to be disambiguated).

[`StableCrateId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/def_id/struct.StableCrateId.html
[`StableCrateId::new`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/def_id/struct.StableCrateId.html#method.new
[`-C metadata`]: https://doc.rust-lang.org/rustc/codegen-options/index.html#metadata

## Crate loading

Crate loading can have quite a few subtle complexities. During [name
resolution], when an external crate is referenced (via an `extern crate` or
path), the resolver uses the [`CrateLoader`] which is responsible for finding
the crate libraries and loading the [metadata] for them. After the dependency
is loaded, the `CrateLoader` will provide the information the resolver needs
to perform its job (such as expanding macros, resolving paths, etc.).

To load each external crate, the `CrateLoader` uses a [`CrateLocator`] to
actually find the correct files for one specific crate. There is some great
documentation in the [`locator`] module that goes into detail on how loading
works, and I strongly suggest reading it to get the full picture.

The location of a dependency can come from several different places. Direct
dependencies are usually passed with `--extern` flags, and the loader can look
at those directly. Direct dependencies often have references to their own
dependencies, which need to be loaded, too. These are usually found by
scanning the directories passed with the `-L` flag for any file whose metadata
contains a matching crate name and [SVH](#strict-version-hash). The loader
will also look at the [sysroot] to find dependencies.

As crates are loaded, they are kept in the [`CStore`] with the crate metadata
wrapped in the [`CrateMetadata`] struct. After resolution and expansion, the
`CStore` will make its way into the [`GlobalCtxt`] for the rest of the
compilation.

[name resolution]: ../name-resolution.md
[`CrateLoader`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/creader/struct.CrateLoader.html
[`CrateLocator`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/locator/struct.CrateLocator.html
[`locator`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/locator/index.html
[`CStore`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/creader/struct.CStore.html
[`CrateMetadata`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/rmeta/decoder/struct.CrateMetadata.html
[`GlobalCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.GlobalCtxt.html
[sysroot]: ../building/bootstrapping/what-bootstrapping-does.md#what-is-a-sysroot

## Pipelining

One trick to improve compile times is to start building a crate as soon as the
metadata for its dependencies is available. For a library, there is no need to
wait for the code generation of dependencies to finish. Cargo implements this
technique by telling `rustc` to emit an [`rmeta`](#rmeta) file for each
dependency as well as an [`rlib`](#rlib). As early as it can, `rustc` will
save the `rmeta` file to disk before it continues to the code generation
phase. The compiler sends a JSON message to let the build tool know that it
can start building the next crate if possible.

The [crate loading](#crate-loading) system is smart enough to know when it
sees an `rmeta` file to use that if the `rlib` is not there (or has only been
partially written).

This pipelining isn't possible for binaries, because the linking phase will
require the code generation of all its dependencies. In the future, it may be
possible to further improve this scenario by splitting linking into a separate
command (see [#64191]).

[#64191]: https://github.com/rust-lang/rust/issues/64191

[metadata]: #metadata
