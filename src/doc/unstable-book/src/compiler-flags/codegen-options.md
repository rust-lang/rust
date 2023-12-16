# Unstable codegen options

All of these options are passed to `rustc` via the `-C` flag, short for "codegen". The flags are
stable but some of their values are individually unstable, and also require using `-Z
unstable-options` to be accepted.

## linker-flavor

In addition to the stable set of linker flavors, the following unstable values also exist:
- `ptx`: use [`rust-ptx-linker`](https://github.com/denzp/rust-ptx-linker)
  for Nvidia NVPTX GPGPU support.
- `bpf`: use [`bpf-linker`](https://github.com/alessandrod/bpf-linker) for eBPF support.

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
