# Unstable codegen options

All of these options are passed to `rustc` via the `-C` flag, short for "codegen". The flags are
stable but some of their values are individually unstable, and also require using `-Z
unstable-options` to be accepted.

## linker-flavor

In addition to the stable set of linker flavors, the following unstable values also exist:
- `ptx`: use [`rust-ptx-linker`](https://github.com/denzp/rust-ptx-linker)
  for Nvidia NVPTX GPGPU support.
- `bpf`: use [`bpf-linker`](https://github.com/alessandrod/bpf-linker) for eBPF support.
- `llbc`: for linking in llvm bitcode. Install the preview rustup components`llvm-bitcode-linker`
  and `llvm-tools` to use as a self-contained linker by passing
  `-Zunstable-options -Clink-self-contained=+linker` together with `-Clinker-flavor=llbc`.
  Can currently only be used for Nvidia NVPTX targets (`nvptx64-nvidia-cuda`).

Additionally, a set of more precise linker flavors also exists, for example allowing targets to
declare that they use the LLD linker by default. The following values are currently unstable, and
the goal is for them to become stable, and preferred in practice over the existing stable values:
- `gnu`: unix-like linker with GNU extensions
- `gnu-lld`: `gnu` using LLD
- `gnu-cc`: `gnu` using a C/C++ compiler as the linker driver
- `gnu-lld-cc`: `gnu` using LLD and a C/C++ compiler as the linker driver
- `darwin`: unix-like linker for Apple targets
- `darwin-lld`: `darwin` using LLD
- `darwin-cc`: `darwin` using a C/C++ compiler as the linker driver
- `darwin-lld-cc`: `darwin` using LLD and a C/C++ compiler as the linker driver
- `wasm-lld`: unix-like linker for Wasm targets, with LLD
- `wasm-lld-cc`: unix-like linker for Wasm targets, with LLD and a C/C++ compiler as the linker
  driver
- `unix`: basic unix-like linker for "any other Unix" targets (Solaris/illumos, L4Re, MSP430, etc),
  not supported with LLD.
- `unix-cc`: `unix` using a C/C++ compiler as the linker driver
- `msvc-lld`: MSVC-style linker for Windows and UEFI, with LLD
- `em-cc`: emscripten compiler frontend, similar to `wasm-lld-cc` with a different interface

## link-self-contained

This flag generally controls whether the linker will use libraries and objects shipped with Rust
instead of those in the system. The stable boolean values for this flag are coarse-grained
(everything or nothing), but there exists a set of unstable values with finer-grained control,
`-Clink-self-contained` can accept a comma-separated list of components, individually enabled
(`+component`) or disabled (`-component`):
- `crto`: CRT objects (e.g. on `windows-gnu`, `musl`, `wasi` targets)
- `libc`: libc static library (e.g. on `musl`, `wasi` targets)
- `unwind`: libgcc/libunwind (e.g. on `windows-gnu`, `fuchsia`, `fortanix`, `gnullvm` targets)
- `linker`: linker, dlltool, and their necessary libraries (e.g. on `windows-gnu` and for
  `rust-lld`)
- `sanitizers`: sanitizer runtime libraries
- `mingw`: other MinGW libs and Windows import libs

Out of the above self-contained linking components, `linker` is the only one currently implemented
(beyond parsing the CLI options) and stabilized.

It refers to the LLD linker, built from the same LLVM revision used by rustc (named `rust-lld` to
avoid naming conflicts), that is distributed via `rustup` with the compiler (and is used by default
for the wasm targets). One can also opt into using it by combining this flag with the appropriate
linker feature: for example, `-Clinker-features=+lld -Clink-self-contained=+linker` will use the
toolchain's `rust-lld` as the linker instead of the system's lld with `-Clinker-features=+lld` only.
