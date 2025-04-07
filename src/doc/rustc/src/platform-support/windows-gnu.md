# \*-windows-gnu

**⚠️ This documentation page is a stub, you can help improving it by sending a PR. ⚠️**

**Tier: 1/2 (with host tools)**

Target triples available:
- `i686-pc-windows-gnu`: Tier 2
- `x86_64-pc-windows-gnu`: Tier 1

## Target maintainers

**⚠️ These targets do not have any maintainers and are not properly maintained. ⚠️**

If you are using this target, consider signing up to become a target maintainer.
See the target tier policy for details.
Without maintainers, these targets may be demoted in the future.

## Requirements

These targets support std and host tools.

Unlike their MSVC counterparts, windows-gnu targets support cross-compilation and are free of all MSVC licensing implications.

They follow Windows calling convention for `extern "C"`.

Like with any other Windows target, created binaries are in PE format.

## Building Rust programs

Rust does ship a pre-compiled std library for those targets.
That means one can easily compile and cross-compile for those targets from other hosts if C proper toolchain is installed.
