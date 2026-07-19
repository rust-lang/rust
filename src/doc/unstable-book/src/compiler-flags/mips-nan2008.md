# `mips-nan2008`

This option enables the LLVM `nan2008` subtarget feature on MIPS targets. It is
only supported on `mips`, `mips32r6`, `mips64`, and `mips64r6` architectures.
Using this flag on other targets is an error.

When enabled, this flag causes the compiler to select the IEEE 754-2008 NaN
encoding for MIPS targets, as opposed to the legacy MIPS NaN encoding. This is
useful when linking against C libraries and toolchains built with GCC's
`-mnan=2008` flag, or when the system ABI mandates the 2008 NaN encoding.

This flag is a target modifier. All crates in the crate graph must agree on
whether the `mips-nan2008` flag is used. The compiler will emit an error if
crates with conflicting settings are linked together. This error can be bypassed
at your own risk using the `-Cunsafe-allow-abi-mismatch=mips-nan2008` flag,
though this may cause runtime failures if the ABIs are truly incompatible.

## Example

To compile code for MIPS with the 2008 NaN encoding:

```sh
rustc -Zmips-nan2008 --target mips-unknown-linux-gnu example.rs
```
