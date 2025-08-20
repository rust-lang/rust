# Codegen Options

All of these options are passed to `rustc` via the `-C` flag, short for "codegen." You can see
a version of this list for your exact compiler by running `rustc -C help`.

## ar

This option is deprecated and does nothing.

## code-model

This option lets you choose which code model to use. \
Code models put constraints on address ranges that the program and its symbols may use. \
With smaller address ranges machine instructions
may be able to use more compact addressing modes.

The specific ranges depend on target architectures and addressing modes available to them. \
For x86 more detailed description of its code models can be found in
[System V Application Binary Interface](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
specification.

Supported values for this option are:

- `tiny` - Tiny code model.
- `small` - Small code model. This is the default model for majority of supported targets.
- `kernel` - Kernel code model.
- `medium` - Medium code model.
- `large` - Large code model.

Supported values can also be discovered by running `rustc --print code-models`.

## codegen-units

This flag controls the maximum number of code generation units the crate is
split into. It takes an integer greater than 0.

When a crate is split into multiple codegen units, LLVM is able to process
them in parallel. Increasing parallelism may speed up compile times, but may
also produce slower code. Setting this to 1 may improve the performance of
generated code, but may be slower to compile.

The default value, if not specified, is 16 for non-incremental builds. For
incremental builds the default is 256 which allows caching to be more granular.

## collapse-macro-debuginfo

This flag controls whether code locations from a macro definition are collapsed into a single
location associated with that macro's call site, when generating debuginfo for this crate.

This option, if passed, overrides both default collapsing behavior and `#[collapse_debuginfo]`
attributes in code.

* `y`, `yes`, `on`, `true`: collapse code locations in debuginfo.
* `n`, `no`, `off` or `false`: do not collapse code locations in debuginfo.
* `external`: collapse code locations in debuginfo only if the macro comes from a different crate.

## control-flow-guard

This flag controls whether LLVM enables the Windows [Control Flow
Guard](https://docs.microsoft.com/en-us/windows/win32/secbp/control-flow-guard)
platform security feature. This flag is currently ignored for non-Windows targets.
It takes one of the following values:

* `y`, `yes`, `on`, `true`, `checks`, or no value: enable Control Flow Guard.
* `nochecks`: emit Control Flow Guard metadata without runtime enforcement checks (this
should only be used for testing purposes as it does not provide security enforcement).
* `n`, `no`, `off`, `false`: do not enable Control Flow Guard (the default).

## debug-assertions

This flag lets you turn `cfg(debug_assertions)` [conditional
compilation](../../reference/conditional-compilation.md#debug_assertions) on
or off. It takes one of the following values:

* `y`, `yes`, `on`, `true`, or no value: enable debug-assertions.
* `n`, `no`, `off` or `false`: disable debug-assertions.

If not specified, debug assertions are automatically enabled only if the
[opt-level](#opt-level) is 0.

## debuginfo

This flag controls the generation of debug information. It takes one of the
following values:

* `0` or `none`: no debug info at all (the default).
* `line-directives-only`: line info directives only. For the nvptx* targets this enables [profiling](https://reviews.llvm.org/D46061). For other use cases, `line-tables-only` is the better, more compatible choice.
* `line-tables-only`: line tables only. Generates the minimal amount of debug info for backtraces with filename/line number info, but not anything else, i.e. no variable or function parameter info.
* `1` or `limited`: debug info without type or variable-level information.
* `2` or `full`: full debug info.

Note: The [`-g` flag][option-g-debug] is an alias for `-C debuginfo=2`.

## default-linker-libraries

This flag controls whether or not the linker includes its default libraries.
It takes one of the following values:

* `y`, `yes`, `on`, `true`: include default libraries.
* `n`, `no`, `off` or `false` or no value: exclude default libraries (the default).

For example, for gcc flavor linkers, this issues the `-nodefaultlibs` flag to
the linker.

## dlltool

On `windows-gnu` targets, this flag controls which dlltool `rustc` invokes to
generate import libraries when using the [`raw-dylib` link kind](../../reference/items/external-blocks.md#the-link-attribute).
It takes a path to [the dlltool executable](https://sourceware.org/binutils/docs/binutils/dlltool.html).
If this flag is not specified, a dlltool executable will be inferred based on
the host environment and target.

## dwarf-version

This option controls the version of DWARF that the compiler emits, on platforms
that use DWARF to encode debug information. It takes one of the following
values:

* `2`: DWARF version 2 (the default on certain platforms, like Android).
* `3`: DWARF version 3 (the default on certain platforms, like AIX).
* `4`: DWARF version 4 (the default on most platforms, like Linux & macOS).
* `5`: DWARF version 5.

DWARF version 1 is not supported.

## embed-bitcode

This flag controls whether or not the compiler embeds LLVM bitcode into object
files. It takes one of the following values:

* `y`, `yes`, `on`, `true` or no value: put bitcode in rlibs (the default).
* `n`, `no`, `off` or `false`: omit bitcode from rlibs.

LLVM bitcode is required when rustc is performing link-time optimization (LTO).
Embedded bitcode will appear in rustc-generated object files inside of a section
whose name is defined by the target platform. Most of the time this is `.llvmbc`.

The use of `-C embed-bitcode=no` can significantly improve compile times and
reduce generated file sizes if your compilation does not actually need bitcode
(e.g. if you're not performing LTO). For these reasons, Cargo uses `-C embed-bitcode=no`
whenever possible. Likewise, if you are building directly with `rustc` we recommend
using `-C embed-bitcode=no` whenever you are not using LTO.

If combined with `-C lto`, `-C embed-bitcode=no` will cause `rustc` to abort
at start-up, because the combination is invalid.

> **Note**: if you're building Rust code with LTO then you probably don't even
> need the `embed-bitcode` option turned on. You'll likely want to use
> `-Clinker-plugin-lto` instead which skips generating object files entirely and
> simply replaces object files with LLVM bitcode. The only purpose for
> `-Cembed-bitcode` is when you're generating an rlib that is both being used
> with and without LTO. For example Rust's standard library ships with embedded
> bitcode since users link to it both with and without LTO.
>
> This also may make you wonder why the default is `yes` for this option. The
> reason for that is that it's how it was for rustc 1.44 and prior. In 1.45 this
> option was added to turn off what had always been the default.

## extra-filename

This option allows you to put extra data in each output filename. It takes a
string to add as a suffix to the filename. See the [`--emit`
flag][option-emit] for more information.

## force-frame-pointers

This flag forces the use of frame pointers. It takes one of the following
values:

* `y`, `yes`, `on`, `true` or no value: force-enable frame pointers.
* `n`, `no`, `off` or `false`: do not force-enable frame pointers. This does
  not necessarily mean frame pointers will be removed.

The default behaviour, if frame pointers are not force-enabled, depends on the
target.

## force-unwind-tables

This flag forces the generation of unwind tables. It takes one of the following
values:

* `y`, `yes`, `on`, `true` or no value: Unwind tables are forced to be generated.
* `n`, `no`, `off` or `false`: Unwind tables are not forced to be generated. If unwind
  tables are required by the target an error will be emitted.

The default if not specified depends on the target.

## incremental

This flag allows you to enable incremental compilation, which allows `rustc`
to save information after compiling a crate to be reused when recompiling the
crate, improving re-compile times. This takes a path to a directory where
incremental files will be stored.

Using incremental compilation inhibits certain optimizations (for example by increasing the amount of codegen units) and is therefore not recommended for release builds.

## inline-threshold

This option is deprecated and does nothing.

Consider using `-Cllvm-args=--inline-threshold=...`.

## instrument-coverage

This option enables instrumentation-based code coverage support. See the
chapter on [instrumentation-based code coverage] for more information.

Note that while the `-C instrument-coverage` option is stable, the profile data
format produced by the resulting instrumentation may change, and may not work
with coverage tools other than those built and shipped with the compiler.

## link-arg

This flag lets you append a single extra argument to the linker invocation.

"Append" is significant; you can pass this flag multiple times to add multiple arguments.

On Unix-like targets that use `cc` as the linker driver, use `-Clink-arg=-Wl,$ARG` to pass an argument to the actual linker.

## link-args

This flag lets you append multiple extra arguments to the linker invocation. The
options should be separated by spaces.

## link-dead-code

Tries to generate and link dead code that would otherwise not be generated or
linked. It takes one of the following values:

* `y`, `yes`, `on`, `true` or no value: try to keep dead code.
* `n`, `no`, `off` or `false`: remove dead code (the default).

This flag was historically used to help improve some older forms of code
coverage measurement. Its use is not recommended.

## link-self-contained

This flag controls whether the linker will use libraries and objects shipped with Rust instead of
those in the system.  It also controls which binary is used for the linker itself. This allows
overriding cases when detection fails or the user wants to use shipped libraries.

You can enable or disable the usage of any self-contained components using one of the following values:

* no value: rustc will use heuristic to disable self-contained mode if system has necessary tools.
* `y`, `yes`, `on`, `true`: use only libraries/objects shipped with Rust.
* `n`, `no`, `off` or `false`: rely on the user or the linker to provide non-Rust libraries/objects.

It is also possible to enable or disable specific self-contained components in a more granular way.
You can pass a comma-separated list of self-contained components, individually enabled
(`+component`) or disabled (`-component`).

Currently, only the `linker` granular option is stabilized, and only on the `x86_64-unknown-linux-gnu` target:
- `linker`: toggle the usage of self-contained linker binaries (linker, dlltool, and their necessary libraries)

Note that only the `-linker` opt-out is stable on the `x86_64-unknown-linux-gnu` target: `+linker` is
already the default on this target.

#### Implementation notes

On the `x86_64-unknown-linux-gnu` target, when using the default linker flavor (using `cc` as the
linker driver) and linker features (to try using `lld`), `rustc` will try to use the self-contained
linker by passing a `-B /path/to/sysroot/` link argument to the driver to find `rust-lld` in the
sysroot. For backwards-compatibility, and to limit name and `PATH` collisions, this is done using a
shim executable (the `lld-wrapper` tool) that forwards execution to the `rust-lld` executable itself.

## linker

This flag controls which linker `rustc` invokes to link your code. It takes a
path to the linker executable. If this flag is not specified, the linker will
be inferred based on the target. See also the [linker-flavor](#linker-flavor)
flag for another way to specify the linker.

Note that on Unix-like targets (for example, `*-unknown-linux-gnu` or `*-unknown-freebsd`)
the C compiler (for example `cc` or `clang`) is used as the "linker" here, serving as a linker driver.
It will invoke the actual linker with all the necessary flags to be able to link against the system libraries like libc.

## linker-features

The `-Clinker-features` flag allows enabling or disabling specific features used during linking.

These feature flags are a flexible extension mechanism that is complementary to linker flavors,
designed to avoid the combinatorial explosion of having to create a new set of flavors for each
linker feature we'd want to use.

The flag accepts a comma-separated list of features, individually enabled (`+feature`) or disabled
(`-feature`).

Currently only one is stable, and only on the `x86_64-unknown-linux-gnu` target:
- `lld`: to toggle trying to use the lld linker, either the system-installed binary, or the self-contained
  `rust-lld` linker (via the [`-Clink-self-contained=+linker`](#link-self-contained) flag).

For example, use:
- `-Clinker-features=+lld` to opt into using the `lld` linker, when possible (see the Implementation notes below)
- `-Clinker-features=-lld` to opt out instead, for targets where it is configured as the default linker

Note that only the `-lld` opt-out is stable on the `x86_64-unknown-linux-gnu` target: `+lld` is
already the default on this target.

#### Implementation notes

On the `x86_64-unknown-linux-gnu` target, when using the default linker flavor (using `cc` as the
linker driver), `rustc` will try to use lld by passing a `-fuse-ld=lld` link argument to the driver.
`rustc` will also try to detect if that _causes_ an error during linking (for example, if GCC is too
old to understand the flag, and returns an error) and will then retry linking without this argument,
as a fallback.

If the user _also_ passes a `-Clink-arg=-fuse-ld=$value`, both will be given to the linker
driver but the user's will be passed last, and would generally have priority over `rustc`'s.

## linker-flavor

This flag controls the linker flavor used by `rustc`. If a linker is given with
the [`-C linker` flag](#linker), then the linker flavor is inferred from the
value provided. If no linker is given then the linker flavor is used to
determine the linker to use. Every `rustc` target defaults to some linker
flavor. Valid options are:

* `em`: use [Emscripten `emcc`](https://emscripten.org/docs/tools_reference/emcc.html).
* `gcc`: use the `cc` executable, which is typically gcc or clang on many systems.
* `ld`: use the `ld` executable.
* `msvc`: use the `link.exe` executable from Microsoft Visual Studio MSVC.
* `wasm-ld`: use the [`wasm-ld`](https://lld.llvm.org/WebAssembly.html)
  executable, a port of LLVM `lld` for WebAssembly.
* `ld64.lld`: use the LLVM `lld` executable with the [`-flavor darwin`
  flag][lld-flavor] for Apple's `ld`.
* `ld.lld`: use the LLVM `lld` executable with the [`-flavor gnu`
  flag][lld-flavor] for GNU binutils' `ld`.
* `lld-link`: use the LLVM `lld` executable with the [`-flavor link`
  flag][lld-flavor] for Microsoft's `link.exe`.

[lld-flavor]: https://releases.llvm.org/12.0.0/tools/lld/docs/Driver.html

## linker-plugin-lto

This flag defers LTO optimizations to the linker. See
[linker-plugin-LTO](../linker-plugin-lto.md) for more details. It takes one of
the following values:

* `y`, `yes`, `on`, `true` or no value: enable linker plugin LTO.
* `n`, `no`, `off` or `false`: disable linker plugin LTO (the default).
* A path to the linker plugin.

More specifically this flag will cause the compiler to replace its typical
object file output with LLVM bitcode files. For example an rlib produced with
`-Clinker-plugin-lto` will still have `*.o` files in it, but they'll all be LLVM
bitcode instead of actual machine code. It is expected that the native platform
linker is capable of loading these LLVM bitcode files and generating code at
link time (typically after performing optimizations).

Note that rustc can also read its own object files produced with
`-Clinker-plugin-lto`. If an rlib is only ever going to get used later with a
`-Clto` compilation then you can pass `-Clinker-plugin-lto` to speed up
compilation and avoid generating object files that aren't used.

## llvm-args

This flag can be used to pass a list of arguments directly to LLVM.

The list must be separated by spaces.

Pass `--help` to see a list of options.

<div class="warning">

Because this flag directly talks to LLVM, it is not subject to the usual stability guarantees of rustc's CLI interface.

</div>

## lto

This flag controls whether LLVM uses [link time
optimizations](https://llvm.org/docs/LinkTimeOptimization.html) to produce
better optimized code, using whole-program analysis, at the cost of longer
linking time. It takes one of the following values:

* `y`, `yes`, `on`, `true`, `fat`, or no value: perform "fat" LTO which attempts to
  perform optimizations across all crates within the dependency graph.
* `thin`: perform ["thin"
  LTO](http://blog.llvm.org/2016/06/thinlto-scalable-and-incremental-lto.html).
  This is similar to "fat", but takes substantially less time to run while
  still achieving performance gains similar to "fat".
  For larger projects like the Rust compiler, ThinLTO can even result in better performance than fat LTO.
* `n`, `no`, `off`, `false`: disables LTO.

If `-C lto` is not specified, then the compiler will attempt to perform "thin
local LTO" which performs "thin" LTO on the local crate only across its
[codegen units](#codegen-units). When `-C lto` is not specified, LTO is
disabled if codegen units is 1 or optimizations are disabled ([`-C
opt-level=0`](#opt-level)). That is:

* When `-C lto` is not specified:
  * `codegen-units=1`: disable LTO.
  * `opt-level=0`: disable LTO.
* When `-C lto` is specified:
  * `lto`: 16 codegen units, perform fat LTO across crates.
  * `codegen-units=1` + `lto`: 1 codegen unit, fat LTO across crates.

See also [linker-plugin-lto](#linker-plugin-lto) for cross-language LTO.

## metadata

This option allows you to control the metadata used for symbol mangling. This
takes a space-separated list of strings. Mangled symbols will incorporate a
hash of the metadata. This may be used, for example, to differentiate symbols
between two different versions of the same crate being linked.

## no-prepopulate-passes

This flag tells the pass manager to use an empty list of passes, instead of the
usual pre-populated list of passes.

When combined with `-O --emit llvm-ir`, it can be used to see the optimized LLVM IR emitted by rustc before any optimizations are applied by LLVM.

## no-redzone

This flag allows you to disable [the
red zone](https://en.wikipedia.org/wiki/Red_zone_\(computing\)). It takes one
of the following values:

* `y`, `yes`, `on`, `true` or no value: disable the red zone.
* `n`, `no`, `off` or `false`: enable the red zone.

The default behaviour, if the flag is not specified, depends on the target.

## no-stack-check

This option is deprecated and does nothing.

## no-vectorize-loops

This flag disables [loop
vectorization](https://llvm.org/docs/Vectorizers.html#the-loop-vectorizer).

## no-vectorize-slp

This flag disables vectorization using
[superword-level
parallelism](https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer).

## opt-level

This flag controls the optimization level.

* `0`: no optimizations, also turns on
  [`cfg(debug_assertions)`](#debug-assertions) (the default).
* `1`: basic optimizations.
* `2`: some optimizations.
* `3`: all optimizations.
* `s`: optimize for binary size.
* `z`: optimize for binary size, but more aggressively. Often results in larger binaries than `s`

Note: The [`-O` flag][option-o-optimize] is an alias for `-C opt-level=3`.

The default is `0`.

## overflow-checks

This flag allows you to control the behavior of [runtime integer
overflow](../../reference/expressions/operator-expr.md#overflow). When
overflow-checks are enabled, a panic will occur on overflow. This flag takes
one of the following values:

* `y`, `yes`, `on`, `true` or no value: enable overflow checks.
* `n`, `no`, `off` or `false`: disable overflow checks.

If not specified, overflow checks are enabled if
[debug-assertions](#debug-assertions) are enabled, disabled otherwise.

## panic

This option lets you control what happens when the code panics.

* `abort`: terminate the process upon panic
* `unwind`: unwind the stack upon panic

If not specified, the default depends on the target.

If any crate in the crate graph uses `abort`, the final binary (`bin`, `dylib`, `cdylib`, `staticlib`) must also use `abort`.
If `std` is used as a `dylib` with `unwind`, the final binary must also use `unwind`.

## passes

This flag can be used to add extra [LLVM
passes](http://llvm.org/docs/Passes.html) to the compilation.

The list must be separated by spaces.

See also the [`no-prepopulate-passes`](#no-prepopulate-passes) flag.

<div class="warning">

Because this flag directly talks to LLVM, it not subject to the usual stability guarantees of rustc's CLI interface.

</div>

## prefer-dynamic

By default, `rustc` prefers to statically link dependencies. This option will
indicate that dynamic linking should be used if possible if both a static and
dynamic versions of a library are available.

There is [an internal algorithm](https://github.com/rust-lang/rust/blob/master/compiler/rustc_metadata/src/dependency_format.rs)
for determining whether or not it is possible to statically or dynamically link
with a dependency.

This flag takes one of the following values:

* `y`, `yes`, `on`, `true` or no value: prefer dynamic linking.
* `n`, `no`, `off` or `false`: prefer static linking (the default).

## profile-generate

This flag allows for creating instrumented binaries that will collect
profiling data for use with profile-guided optimization (PGO). The flag takes
an optional argument which is the path to a directory into which the
instrumented binary will emit the collected data. See the chapter on
[profile-guided optimization] for more information.

## profile-use

This flag specifies the profiling data file to be used for profile-guided
optimization (PGO). The flag takes a mandatory argument which is the path
to a valid `.profdata` file. See the chapter on
[profile-guided optimization] for more information.

## relocation-model

This option controls generation of
[position-independent code (PIC)](https://en.wikipedia.org/wiki/Position-independent_code).

Supported values for this option are:

#### Primary relocation models

- `static` - non-relocatable code, machine instructions may use absolute addressing modes.

- `pic` - fully relocatable position independent code,
machine instructions need to use relative addressing modes.  \
Equivalent to the "uppercase" `-fPIC` or `-fPIE` options in other compilers,
depending on the produced crate types.  \
This is the default model for majority of supported targets.

- `pie` - position independent executable, relocatable code but without support for symbol
interpositioning (replacing symbols by name using `LD_PRELOAD` and similar). Equivalent to the "uppercase" `-fPIE` option in other compilers. `pie`
code cannot be linked into shared libraries (you'll get a linking error on attempt to do this).

#### Special relocation models

- `dynamic-no-pic` - relocatable external references, non-relocatable code.  \
Only makes sense on Darwin and is rarely used.  \
If StackOverflow tells you to use this as an opt-out of PIC or PIE, don't believe it,
use `-C relocation-model=static` instead.
- `ropi`, `rwpi` and `ropi-rwpi` - relocatable code and read-only data, relocatable read-write data,
and combination of both, respectively.  \
Only makes sense for certain embedded ARM targets.
- `default` - relocation model default to the current target.  \
Only makes sense as an override for some other explicitly specified relocation model
previously set on the command line.

Supported values can also be discovered by running `rustc --print relocation-models`.

#### Linking effects

In addition to codegen effects, `relocation-model` has effects during linking.

If the relocation model is `pic` and the current target supports position-independent executables
(PIE), the linker will be instructed (`-pie`) to produce one.  \
If the target doesn't support both position-independent and statically linked executables,
then `-C target-feature=+crt-static` "wins" over `-C relocation-model=pic`,
and the linker is instructed (`-static`) to produce a statically linked
but not position-independent executable.

## relro-level

This flag controls what level of RELRO (Relocation Read-Only) is enabled. RELRO is an exploit
mitigation which makes the Global Offset Table (GOT) read-only.

Supported values for this option are:

- `off`: Dynamically linked functions are resolved lazily and the GOT is writable.
- `partial`: Dynamically linked functions are resolved lazily and written into the Procedure
  Linking Table (PLT) part of the GOT (`.got.plt`). The non-PLT part of the GOT (`.got`) is made
  read-only and both are moved to prevent writing from buffer overflows.
- `full`: Dynamically linked functions are resolved at the start of program execution and the
  Global Offset Table (`.got`/`.got.plt`) is populated eagerly and then made read-only. The GOT is
  also moved to prevent writing from buffer overflows. Full RELRO uses more memory and increases
  process startup time.

This flag is ignored on platforms where RELRO is not supported (targets which do not use the ELF
binary format), such as Windows or macOS. Each rustc target has its own default for RELRO. rustc
enables Full RELRO by default on platforms where it is supported.

## remark

This flag lets you print remarks for optimization passes.

The list of passes should be separated by spaces.

`all` will remark on every pass.

## rpath

This flag controls whether rustc sets an [`rpath`](https://en.wikipedia.org/wiki/Rpath) for the binary.
It takes one of the following values:

* `y`, `yes`, `on`, `true` or no value: enable rpath.
* `n`, `no`, `off` or `false`: disable rpath (the default).

This flag only does something on Unix-like platforms (Mach-O and ELF), it is ignored on other platforms.

If enabled, rustc will add output-relative (using `@load_path` on Mach-O and `$ORIGIN` on ELF respectively) rpaths to all `dylib` dependencies.

For example, for the following directory structure, with `libdep.so` being a `dylib` crate compiled with `-Cprefer-dynamic`:

```text
dep
 |- libdep.so
a.rs
```

`rustc a.rs --extern dep=dep/libdep.so -Crpath` will, on x86-64 Linux, result in approximately the following `DT_RUNPATH`: `$ORIGIN/dep:$ORIGIN/$RELATIVE_PATH_TO_SYSROOT/lib/rustlib/x86_64-unknown-linux-gnu/lib` (where `RELATIVE_PATH_TO_SYSROOT` depends on the build directory location).

This is primarily useful for local development, to ensure that all the `dylib` dependencies can be found appropriately.

To set the rpath to a different value (which can be useful for distribution), `-Clink-arg` with a platform-specific linker argument can be used to set the rpath directly.

## save-temps

This flag controls whether temporary files generated during compilation are
deleted once compilation finishes. It takes one of the following values:

* `y`, `yes`, `on`, `true` or no value: save temporary files.
* `n`, `no`, `off` or `false`: delete temporary files (the default).

## soft-float

This option controls whether `rustc` generates code that emulates floating
point instructions in software. It takes one of the following values:

* `y`, `yes`, `on`, `true` or no value: use soft floats.
* `n`, `no`, `off` or `false`: use hardware floats (the default).

This flag only works on `*eabihf` targets and **is unsound and deprecated**.

## split-debuginfo

This option controls the emission of "split debuginfo" for debug information
that `rustc` generates. The default behavior of this option is
platform-specific, and not all possible values for this option work on all
platforms. Possible values are:

* `off` - This is the default for platforms with ELF binaries and windows-gnu
  (not Windows MSVC and not macOS). This typically means that DWARF debug
  information can be found in the final artifact in sections of the executable.
  This option is not supported on Windows MSVC. On macOS this options prevents
  the final execution of `dsymutil` to generate debuginfo.

* `packed` - This is the default for Windows MSVC and macOS. The term
  "packed" here means that all the debug information is packed into a separate
  file from the main executable. On Windows MSVC this is a `*.pdb` file, on
  macOS this is a `*.dSYM` folder, and on other platforms this is a `*.dwp`
  file.

* `unpacked` - This means that debug information will be found in separate
  files for each compilation unit (object file). This is not supported on
  Windows MSVC. On macOS this means the original object files will contain
  debug information. On other Unix platforms this means that `*.dwo` files will
  contain debug information.

Note that all three options are supported on Linux and Apple platforms,
`packed` is supported on Windows-MSVC, and all other platforms support `off`.
Attempting to use an unsupported option requires using the nightly channel
with the `-Z unstable-options` flag.

## strip

The option `-C strip=val` controls stripping of debuginfo and similar auxiliary
data from binaries during linking.

Supported values for this option are:

- `none` - debuginfo and symbols are not modified.
- `debuginfo` - debuginfo sections and debuginfo symbols from the symbol table
  section are stripped at link time and are not copied to the produced binary.
  This should leave backtraces mostly-intact but may make using a debugger like
  gdb or lldb ineffectual. Prior to 1.79, this unintentionally disabled the
  generation of `*.pdb` files on MSVC, resulting in the absence of symbols.
- `symbols` - same as `debuginfo`, but the rest of the symbol table section is
  stripped as well, depending on platform support. On platforms which depend on
  this symbol table for backtraces, profiling, and similar, this can affect
  them so negatively as to make the trace incomprehensible. Programs which may
  be combined with others, such as CLI pipelines and developer tooling, or even
  anything which wants crash-reporting, should usually avoid `-Cstrip=symbols`.

Note that, at any level, removing debuginfo only necessarily impacts "friendly"
introspection. `-Cstrip` cannot be relied on as a meaningful security or
obfuscation measure, as disassemblers and decompilers can extract considerable
information even in the absence of symbols.

## symbol-mangling-version

This option controls the [name mangling] format for encoding Rust item names
for the purpose of generating object code and linking.

Supported values for this option are:

* `v0` — The "v0" mangling scheme.

The default, if not specified, will use a compiler-chosen default which may
change in the future.

See the [Symbol Mangling] chapter for details on symbol mangling and the mangling format.

[name mangling]: https://en.wikipedia.org/wiki/Name_mangling
[Symbol Mangling]: ../symbol-mangling/index.md

## target-cpu

This instructs `rustc` to generate code specifically for a particular processor.

You can run `rustc --print target-cpus` to see the valid options to pass
and the default target CPU for the current build target.
Each target has a default base CPU. Special values include:

* `native` can be passed to use the processor of the host machine.
* `generic` refers to an LLVM target with minimal features but modern tuning.

## target-feature

Individual targets will support different features; this flag lets you control
enabling or disabling a feature. Each feature should be prefixed with a `+` to
enable it or `-` to disable it.

Features from multiple `-C target-feature` options are combined. \
Multiple features can be specified in a single option by separating them
with commas - `-C target-feature=+x,-y`. \
If some feature is specified more than once with both `+` and `-`,
then values passed later override values passed earlier. \
For example, `-C target-feature=+x,-y,+z -Ctarget-feature=-x,+y`
is equivalent to `-C target-feature=-x,+y,+z`.

To see the valid options and an example of use, run `rustc --print
target-features`.

Using this flag is unsafe and might result in [undefined runtime
behavior](../targets/known-issues.md).

See also the [`target_feature`
attribute](../../reference/attributes/codegen.md#the-target_feature-attribute)
for controlling features per-function.

This also supports the feature `+crt-static` and `-crt-static` to control
[static C runtime linkage](../../reference/linkage.html#static-and-dynamic-c-runtimes).

Each target and [`target-cpu`](#target-cpu) has a default set of enabled
features.

## tune-cpu

This instructs `rustc` to schedule code specifically for a particular
processor. This does not affect the compatibility (instruction sets or ABI),
but should make your code slightly more efficient on the selected CPU.

The valid options are the same as those for [`target-cpu`](#target-cpu).
The default is `None`, which LLVM translates as the `target-cpu`.

This is an unstable option. Use `-Z tune-cpu=machine` to specify a value.

Due to limitations in LLVM (12.0.0-git9218f92), this option is currently
effective only for x86 targets.

[option-emit]: ../command-line-arguments.md#option-emit
[option-o-optimize]: ../command-line-arguments.md#option-o-optimize
[instrumentation-based code coverage]: ../instrument-coverage.md
[profile-guided optimization]: ../profile-guided-optimization.md
[option-g-debug]: ../command-line-arguments.md#option-g-debug
