# Codegen options

All of these options are passed to `rustc` via the `-C` flag, short for "codegen." You can see
a version of this list for your exact compiler by running `rustc -C help`.

## ar

This option is deprecated and does nothing.

## linker

This flag lets you control which linker `rustc` invokes to link your code. It
takes a path to the linker executable. If this flag is not specified, the
linker will be inferred based on the target. See also the
[linker-flavor](#linker-flavor) flag for another way to specify the linker.

## link-arg

This flag lets you append a single extra argument to the linker invocation.

"Append" is significant; you can pass this flag multiple times to add multiple arguments.

## link-args

This flag lets you append multiple extra arguments to the linker invocation. The
options should be separated by spaces.

## linker-flavor

This flag lets you control the linker flavor used by `rustc`. If a linker is given with the
[`-C linker` flag](#linker), then the linker flavor is inferred from the value provided. If no
linker is given then the linker flavor is used to determine the linker to use. Every `rustc` target
defaults to some linker flavor. Valid options are:

* `em`: Uses [Emscripten `emcc`](https://emscripten.org/docs/tools_reference/emcc.html).
* `gcc`: Uses the `cc` executable, which is typically gcc or clang on many systems.
* `ld`: Uses the `ld` executable.
* `msvc`: Uses the `link.exe` executable from Microsoft Visual Studio MSVC.
* `ptx-linker`: Uses
  [`rust-ptx-linker`](https://github.com/denzp/rust-ptx-linker) for Nvidia
  NVPTX GPGPU support.
* `wasm-ld`: Uses the [`wasm-ld`](https://lld.llvm.org/WebAssembly.html)
  executable, a port of LLVM `lld` for WebAssembly.
* `ld64.lld`: Uses the LLVM `lld` executable with the [`-flavor darwin`
  flag][lld-flavor] for Apple's `ld`.
* `ld.lld`: Uses the LLVM `lld` executable with the [`-flavor gnu`
  flag][lld-flavor] for GNU binutils' `ld`.
* `lld-link`: Uses the LLVM `lld` executable with the [`-flavor link`
  flag][lld-flavor] for Microsoft's `link.exe`.

[lld-flavor]: https://lld.llvm.org/Driver.html

## link-dead-code

Normally, the linker will remove dead code. This flag disables this behavior.

An example of when this flag might be useful is when trying to construct code coverage
metrics.

## lto

This flag instructs LLVM to use [link time
optimizations](https://llvm.org/docs/LinkTimeOptimization.html) to produce
better optimized code, using whole-program analysis, at the cost of longer
linking time.

This flag may take one of the following values:

* `y`, `yes`, `on`, `fat`, or no value: Performs "fat" LTO which attempts to
  perform optimizations across all crates within the dependency graph.
* `n`, `no`, `off`: Disables LTO.
* `thin`: Performs ["thin"
  LTO](http://blog.llvm.org/2016/06/thinlto-scalable-and-incremental-lto.html).
  This is similar to "fat", but takes substantially less time to run while
  still achieving performance gains similar to "fat".

If `-C lto` is not specified, then the compiler will attempt to perform "thin
local LTO" which performs "thin" LTO on the local crate only across its
[codegen units](#codegen-units). When `-C lto` is not specified, LTO is
disabled if codegen units is 1 or optimizations are disabled ([`-C
opt-level=0`](#opt-level)). That is:

* When `-C lto` is not specified:
  * `codegen-units=1`: Disables LTO.
  * `opt-level=0`: Disables LTO.
* When `-C lto=true`:
  * `lto=true`: 16 codegen units, performs fat LTO across crates.
  * `codegen-units=1` + `lto=true`: 1 codegen unit, fat LTO across crates.

See also [linker-plugin-lto](#linker-plugin-lto) for cross-language LTO.

## linker-plugin-lto

Defers LTO optimizations to the linker. See
[linkger-plugin-LTO](../linker-plugin-lto.md) for more details. Takes one of
the following values:

* `y`, `yes`, `on`, or no value: Enabled.
* `n`, `no`, or `off`: Disabled (default).
* A path to the linker plugin.

## target-cpu

This instructs `rustc` to generate code specifically for a particular processor.

You can run `rustc --print target-cpus` to see the valid options to pass
here. Additionally, `native` can be passed to use the processor of the host
machine. Each target has a default base CPU.

## target-feature

Individual targets will support different features; this flag lets you control
enabling or disabling a feature. Each feature should be prefixed with a `+` to
enable it or `-` to disable it. Separate multiple features with commas.

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

## passes

This flag can be used to add extra [LLVM
passes](http://llvm.org/docs/Passes.html) to the compilation.

The list must be separated by spaces.

See also the [`no-prepopulate-passes`](#no-prepopulate-passes) flag.

## llvm-args

This flag can be used to pass a list of arguments directly to LLVM.

The list must be separated by spaces.

Pass `--help` to see a list of options.

## save-temps

`rustc` will generate temporary files during compilation; normally it will
delete them after it's done with its work. This option will cause them to be
preserved instead of removed.

## rpath

This option allows you to enable
[`rpath`](https://en.wikipedia.org/wiki/Rpath).

## overflow-checks

This flag allows you to control the behavior of [runtime integer
overflow](../../reference/expressions/operator-expr.md#overflow). When
overflow-checks are enabled, a panic will occur on overflow. This flag takes
one of the following values:

* `y`, `yes`, `on`, or no value: Enable overflow checks.
* `n`, `no`, or `off`: Disable overflow checks.

If not specified, overflow checks are enabled if
[debug-assertions](#debug-assertions) are enabled, disabled otherwise.

## no-prepopulate-passes

The pass manager comes pre-populated with a list of passes; this flag
ensures that list is empty.

## no-vectorize-loops

By default, `rustc` will attempt to [vectorize
loops](https://llvm.org/docs/Vectorizers.html#the-loop-vectorizer). This
flag will turn that behavior off.

## no-vectorize-slp

By default, `rustc` will attempt to vectorize code using [superword-level
parallelism](https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer). This
flag will turn that behavior off.

## soft-float

This option will make `rustc` generate code using "soft floats." By default,
a lot of hardware supports floating point instructions, and so the code generated
will take advantage of this. "soft floats" emulate floating point instructions
in software.

## prefer-dynamic

By default, `rustc` prefers to statically link dependencies. This option will
indicate that dynamic linking should be used if possible if both a static and
dynamic versions of a library are available. There is an internal algorithm
for determining whether or not it is possible to statically or dynamically
link with a dependency. For example, `cdylib` crate types may only use static
linkage.

## no-integrated-as

`rustc` normally uses the LLVM internal assembler to create object code. This
flag will disable the internal assembler and emit assembly code to be
translated using an external assembler, currently the linker such as `cc`.

## no-redzone

This flag allows you to disable [the
red zone](https://en.wikipedia.org/wiki/Red_zone_\(computing\)). This flag can
be passed one of the following options:

* `y`, `yes`, `on`, or no value: Disables the red zone.
* `n`, `no`, or `off`: Enables the red zone.

The default if not specified depends on the target.

## relocation-model

This option lets you choose which
[relocation](https://en.wikipedia.org/wiki/Relocation_\(computing\)) model to
use.

To find the valid options for this flag, run `rustc --print relocation-models`.

## code-model

This option lets you choose which code model to use.

To find the valid options for this flag, run `rustc --print code-models`.

## metadata

This option allows you to control the metadata used for symbol mangling. This
takes a space-separated list of strings. Mangled symbols will incorporate a
hash of the metadata. This may be used, for example, to differentiate symbols
between two different versions of the same crate being linked.

## extra-filename

This option allows you to put extra data in each output filename. It takes a
string to add as a suffix to the filename. See the [`--emit`
flag][option-emit] for more information.

## codegen-units

This flag controls how many code generation units the crate is split into. It
takes an integer greater than 0.

When a crate is split into multiple codegen units, LLVM is able to process
them in parallel. Increasing parallelism may speed up compile times, but may
also produce slower code. Setting this to 1 may improve the performance of
generated code, but may be slower to compile.

The default, if not specified, is 16. This flag is ignored if
[incremental](#incremental) is enabled, in which case an internal heuristic is
used to split the crate.

## remark

This flag lets you print remarks for optimization passes.

The list of passes should be separated by spaces.

`all` will remark on every pass.

## no-stack-check

This option is deprecated and does nothing.

## debuginfo

This flag lets you control debug information:

* `0`: no debug info at all (default)
* `1`: line tables only
* `2`: full debug info

Note: The [`-g` flag][option-g-debug] is an alias for `-C debuginfo=2`.

## opt-level

This flag lets you control the optimization level.

* `0`: no optimizations, also turns on [`cfg(debug_assertions)`](#debug-assertions).
* `1`: basic optimizations
* `2`: some optimizations
* `3`: all optimizations
* `s`: optimize for binary size
* `z`: optimize for binary size, but also turn off loop vectorization.

Note: The [`-O` flag][option-o-optimize] is an alias for `-C opt-level=2`.

The default is `0`.

## debug-assertions

This flag lets you turn `cfg(debug_assertions)` [conditional
compilation](../../reference/conditional-compilation.md#debug_assertions) on
or off. It takes one of the following values:

* `y`, `yes`, `on`, or no value: Enable debug-assertions.
* `n`, `no`, or `off`: Disable debug-assertions.

If not specified, debug assertions are automatically enabled only if the
[opt-level](#opt-level) is 0.

## inline-threshold

This option lets you set the default threshold for inlining a function. It
takes an unsigned integer as a value. Inlining is based on a cost model, where
a higher threshold will allow more inlining.

The default depends on the [opt-level](#opt-level):

| opt-level | Threshold |
|-----------|-----------|
| 0         | N/A, only inlines always-inline functions |
| 1         | N/A, only inlines always-inline functions and LLVM lifetime intrinsics |
| 2         | 225 |
| 3         | 275 |
| s         | 75 |
| z         | 25 |

## panic

This option lets you control what happens when the code panics.

* `abort`: terminate the process upon panic
* `unwind`: unwind the stack upon panic

If not specified, the default depends on the target.

## incremental

This flag allows you to enable incremental compilation, which allows `rustc`
to save information after compiling a crate to be reused when recompiling the
crate, improving re-compile times. This takes a path to a directory where
incremental files will be stored.

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

## force-frame-pointers

This flag forces the use of frame pointers. It takes one of the following
values:

* `y`, `yes`, `on`, or no value: Frame pointers are forced to be enabled.
* `n`, `no`, or `off`: Frame pointers are not forced to be enabled. This does
  not necessarily mean frame pointers will be removed.

The default if not specified depends on the target.

## default-linker-libraries

This flag controls whether or not the linker includes its default libraries.
It takes one of the following values:

* `y`, `yes`, `on`, or no value: Default libraries are included.
* `n`, `no`, or `off`: Default libraries are **not** included.

For example, for gcc flavor linkers, this issues the `-nodefaultlibs` flag to
the linker.

The default is `yes` if not specified.

[option-emit]: ../command-line-arguments.md#option-emit
[option-o-optimize]: ../command-line-arguments.md#option-o-optimize
[profile-guided optimization]: ../profile-guided-optimization.md
[option-g-debug]: ../command-line-arguments.md#option-g-debug
