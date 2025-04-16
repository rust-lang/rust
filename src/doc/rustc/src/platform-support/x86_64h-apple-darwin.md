# `x86_64h-apple-darwin`

**Tier: 3**

Target for macOS on late-generation `x86_64` Apple chips, usable as the
`x86_64h` entry in universal binaries, and equivalent to LLVM's
`x86_64h-apple-macosx*` targets.

## Target maintainers

- Thom Chiovoloni `thom@shift.click` <https://github.com/thomcc>

## Requirements

This target is an `x86_64` target that only supports Apple's late-gen
(Haswell-compatible) Intel chips. It enables a set of target features available
on these chips (AVX2 and similar), and MachO binaries built with this target may
be used as the `x86_64h` entry in universal binaries ("fat" MachO binaries), and
will fail to load on machines that do not support this.

It should support the full standard library (`std` and `alloc` either with
default or user-defined allocators). This target is probably most useful when
targeted via cross-compilation (including from `x86_64-apple-darwin`), but if
built manually, the host tools work.

It is similar to [`x86_64-apple-darwin`](apple-darwin.md) in nearly all
respects.

## Building the target

Users on Apple targets can build this by adding it to the `target` list in
`bootstrap.toml`, or with `-Zbuild-std`.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Code built with this target can be run on the set of Intel macOS machines that
support running `x86_64h` binaries (relatively recent Intel macs). The Rust test
suite seems to work.

## Cross-compilation toolchains and C code

Cross-compilation to this target from Apple hosts should generally work without
much configuration, so long as XCode and the CommandLineTools are installed.
Targeting it from non-Apple hosts is difficult, but no more so than targeting
`x86_64-apple-darwin`.

When compiling C code for this target, either the "`x86_64h-apple-macosx*`" LLVM
targets should be used, or an argument like `-arch x86_64h` should be passed to
the C compiler.
