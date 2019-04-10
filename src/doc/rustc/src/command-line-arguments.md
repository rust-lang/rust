# Command-line arguments

Here's a list of command-line arguments to `rustc` and what they do.

## `-h`/`--help`: get help

This flag will print out help information for `rustc`.

## `--cfg`: configure the compilation environment

This flag can turn on or off various `#[cfg]` settings.

The value can either be a single identifier or two identifiers separated by `=`.

For examples, `--cfg 'verbose'` or `--cfg 'feature="serde"'`. These correspond
to `#[cfg(verbose)]` and `#[cfg(feature = "serde")]` respectively.

## `-L`: add a directory to the library search path

When looking for external crates, a directory passed to this flag will be searched.

## `-l`: link the generated crate to a native library

This flag allows you to specify linking to a specific native library when building
a crate.

## `--crate-type`: a list of types of crates for the compiler to emit

This instructs `rustc` on which crate type to build.

## `--crate-name`: specify the name of the crate being built

This informs `rustc` of the name of your crate.

## `--emit`: emit output other than a crate

Instead of producing a crate, this flag can print out things like the assembly or LLVM-IR.

## `--print`: print compiler information

This flag prints out various information about the compiler.

## `-g`: include debug information

A synonym for `-C debuginfo=2`, for more see [here](codegen-options/index.html#debuginfo).

## `-O`: optimize your code

A synonym for `-C opt-level=2`, for more see [here](codegen-options/index.html#opt-level).

## `-o`: filename of the output

This flag controls the output filename.

## `--out-dir`: directory to write the output in

The outputted crate will be written to this directory.

## `--explain`: provide a detailed explanation of an error message

Each error of `rustc`'s comes with an error code; this will print
out a longer explanation of a given error.

## `--test`: build a test harness

When compiling this crate, `rustc` will ignore your `main` function
and instead produce a test harness.

## `--target`: select a target triple to build

This controls which [target](targets/index.html) to produce.

## `-W`: set lint warnings

This flag will set which lints should be set to the [warn level](lints/levels.html#warn).

## `-A`: set lint allowed

This flag will set which lints should be set to the [allow level](lints/levels.html#allow).

## `-D`: set lint denied

This flag will set which lints should be set to the [deny level](lints/levels.html#deny).

## `-F`: set lint forbidden

This flag will set which lints should be set to the [forbid level](lints/levels.html#forbid).

## `-Z`: set unstable options

This flag will allow you to set unstable options of rustc. In order to set multiple options,
the -Z flag can be used multiple times. For example: `rustc -Z verbose -Z time`.
Specifying options with -Z is only available on nightly. To view all available options
run: `rustc -Z help`.

## `--cap-lints`: set the most restrictive lint level

This flag lets you 'cap' lints, for more, [see here](lints/levels.html#capping-lints).

## `-C`/`--codegen`: code generation options

This flag will allow you to set [codegen options](codegen-options/index.html).

## `-V`/`--version`: print a version

This flag will print out `rustc`'s version.

## `-v`/`--verbose`: use verbose output

This flag, when combined with other flags, makes them produce extra output.

## `--extern`: specify where an external library is located

This flag allows you to pass the name and location of an external crate that will
be linked into the crate you're buildling.

## `--sysroot`: Override the system root

The "sysroot" is where `rustc` looks for the crates that come with the Rust
distribution; this flag allows that to be overridden.

## `--error-format`: control how errors are produced

This flag lets you control the format of errors.

## `--color`: configure coloring of output

This flag lets you control color settings of the output.
