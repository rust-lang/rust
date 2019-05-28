# Codegen options

All of these options are passed to `rustc` via the `-C` flag, short for "codegen." You can see
a version of this list for your exact compiler by running `rustc -C help`.

## ar

This option is deprecated and does nothing.

## linker

This flag lets you control which linker `rustc` invokes to link your code.

## link-arg=val

This flag lets you append a single extra argument to the linker invocation.

"Append" is significant; you can pass this flag multiple times to add multiple arguments.

## link-args

This flag lets you append multiple extra arguments to the linker invocation. The
options should be separated by spaces.

## linker-flavor

This flag lets you control the linker flavor used by `rustc`. If a linker is given with the
`-C linker` flag described above then the linker flavor is inferred from the value provided. If no
linker is given then the linker flavor is used to determine the linker to use. Every `rustc` target
defaults to some linker flavor.

## link-dead-code

Normally, the linker will remove dead code. This flag disables this behavior.

An example of when this flag might be useful is when trying to construct code coverage
metrics.

## lto

This flag instructs LLVM to use [link time
optimizations](https://llvm.org/docs/LinkTimeOptimization.html).

It takes one of two values, `thin` and `fat`. 'thin' LTO [is a new feature of
LLVM](http://blog.llvm.org/2016/06/thinlto-scalable-and-incremental-lto.html),
'fat' referring to the classic version of LTO.

## target-cpu

This instructs `rustc` to generate code specifically for a particular processor.

You can run `rustc --print target-cpus` to see the valid options to pass
here. Additionally, `native` can be passed to use the processor of the host
machine.

## target-feature

Individual targets will support different features; this flag lets you control
enabling or disabling a feature.

To see the valid options and an example of use, run `rustc --print
target-features`.

## passes

This flag can be used to add extra LLVM passes to the compilation.

The list must be separated by spaces.

## llvm-args

This flag can be used to pass a list of arguments directly to LLVM.

The list must be separated by spaces.

## save-temps

`rustc` will generate temporary files during compilation; normally it will
delete them after it's done with its work. This option will cause them to be
preserved instead of removed.

## rpath

This option allows you to set the value of
[`rpath`](https://en.wikipedia.org/wiki/Rpath).

## overflow-checks

This flag allows you to control the behavior of integer overflow. This flag
can be passed many options:

* To turn overflow checks on: `y`, `yes`, or `on`.
* To turn overflow checks off: `n`, `no`, or `off`.

## no-prepopulate-passes

The pass manager comes pre-populated with a list of passes; this flag
ensures that list is empty.

## no-vectorize-loops

By default, `rustc` will attempt to [vectorize
loops](https://llvm.org/docs/Vectorizers.html#the-loop-vectorizer). This
flag will turn that behavior off.

## no-vectorize-slp

By default, `rustc` will attempt to vectorize loops using [superword-level
parallelism](https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer). This
flag will turn that behavior off.

## soft-float

This option will make `rustc` generate code using "soft floats." By default,
a lot of hardware supports floating point instructions, and so the code generated
will take advantage of this. "soft floats" emulate floating point instructions
in software.

## prefer-dynamic

By default, `rustc` prefers to statically link dependencies. This option will
make it use dynamic linking instead.

## no-integrated-as

LLVM comes with an internal assembler; this option will let you use an
external assembler instead.

## no-redzone

This flag allows you to disable [the
red zone](https://en.wikipedia.org/wiki/Red_zone_\(computing\)). This flag can
be passed many options:

* To enable the red zone: `y`, `yes`, or `on`.
* To disable it: `n`, `no`, or `off`.

## relocation-model

This option lets you choose which relocation model to use.

To find the valid options for this flag, run `rustc --print relocation-models`.

## code-model=val

This option lets you choose which code model to use.

To find the valid options for this flag, run `rustc --print code-models`.

## metadata

This option allows you to control the metadata used for symbol mangling.

## extra-filename

This option allows you to put extra data in each output filename.

## codegen-units

This flag lets you control how many threads are used when doing
code generation.

Increasing parallelism may speed up compile times, but may also
produce slower code.

## remark

This flag lets you print remarks for these optimization passes.

The list of passes should be separated by spaces.

`all` will remark on every pass.

## no-stack-check

This option is deprecated and does nothing.

## debuginfo

This flag lets you control debug information:

* `0`: no debug info at all
* `1`: line tables only
* `2`: full debug info

## opt-level

This flag lets you control the optimization level.

* `0`: no optimizations, also turn on `cfg(debug_assertions)`.
* `1`: basic optimizations
* `2`: some optimizations
* `3`: all optimizations
* `s`: optimize for binary size
* `z`: optimize for binary size, but also turn off loop vectorization.

## debug-assertions

This flag lets you turn `cfg(debug_assertions)` on or off.

## inline-threshold

This option lets you set the threshold for inlining a function.

The default is 225.

## panic

This option lets you control what happens when the code panics.

* `abort`: terminate the process upon panic
* `unwind`: unwind the stack upon panic

## incremental

This flag allows you to enable incremental compilation.

## profile-generate

This flag allows for creating instrumented binaries that will collect
profiling data for use with profile-guided optimization (PGO). The flag takes
an optional argument which is the path to a directory into which the
instrumented binary will emit the collected data. See the chapter on
[profile-guided optimization](profile-guided-optimization.html) for more
information.

## profile-use

This flag specifies the profiling data file to be used for profile-guided
optimization (PGO). The flag takes a mandatory argument which is the path
to a valid `.profdata` file. See the chapter on
[profile-guided optimization](profile-guided-optimization.html) for more
information.
