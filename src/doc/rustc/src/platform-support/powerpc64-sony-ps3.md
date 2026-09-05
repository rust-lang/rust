# `powerpc64-sony-ps3`

**Tier: 3**

Target for the Sony PlayStation 3 (shortened to "PS3"), for the PowerPC Processor Element (PPU) of the [Cell Broadband Engine Architecture (CBEA)](https://ieeexplore.ieee.org/document/5388675).

## Target maintainers

- [@ZephyrCodesStuff](https://github.com/ZephyrCodesStuff) (Primary developer and maintainer)
- [@RipleyTom](https://github.com/RipleyTom) (Fallback maintainer)

## Requirements

The target is a **big-endian PowerPC64 ELFv1** platform (the Cell Broadband Engine's PPE), and intended only for use on Sony PlayStation 3 systems, under the official operating system, "CellOS".

The linker must support **Big-Endian PowerPC64 ELFv1**: the recommended and tested linker is [mold](https://github.com/rui314/mold). LLVM's `lld` does not correctly handle ELFv1 call relocations in freestanding `no_std` environments, making it incompatible. (See: [rust-lang/rust#85589](https://github.com/rust-lang/rust/issues/85589), [llvm/llvm-project#27630](https://github.com/llvm/llvm-project/issues/27630))

Resulting binaries require additional patching after linking to adhere to the PlayStation 3 operating system, in order to be bootable. An open-source patcher is available [here](https://github.com/ZephyrCodesStuff/rust-ps3/tree/main/moldier). Generally, a patcher must perform the following:

- Rewrite the ELF OS/ABI to `0x66` (`ELFOSABI_CELLLV2`)
- Strip any GNU/Linux headers
- Add Sony-specific flags, sections (`.sys_proc_param` and `.sys_proc_prx_param`) and headers
- Add "stubs" for Sony SPRX dynamic-link libraries, by adding a section (`.lib.stub`) for CellOS to be able to link them
- Patch OPD function descriptors (in the `.opd` section)

_**Note**: this list may not be exhaustive for all use cases, but is sufficient for producing a runnable binary. Producing a PRX dynamic library may require more/different steps._


The target _fully supports_:

- The Rust `core` features
- The Rust `alloc` feature, as the CellOS Lv2 kernel provides virtual memory allocation (`sys_memory_allocate`) on top of which a heap allocator (such as [talc](https://github.com/SFBdragon/talc)) can be implemented.
- AltiVec / VMX SIMD vector extensions (natively supported by LLVM via `+altivec`)

## Building the target

If `rustc` is built with this target enabled, no external C cross-compilation toolchain is strictly required to build the compiler host artifacts, but `mold` must be installed on the host system to perform linking.

Support for using `lld` as a linker is unlikely, until support for ELFv1 is implemented on `lld`.

## Building Rust programs

Because this is a Tier 3 target, pre-compiled standard library artifacts (`core`, `alloc`) are not distributed via rustup. Programs must be built using a nightly toolchain with the `rust-src` component and `-Z build-std`.

A Rust SDK ready for development exists open-sourced [here](https://github.com/Zephyrcodesstuff/rust-ps3) and is licensed `MIT OR Apache-2.0`.

Configure your project `.cargo/config.toml`:
```toml
[target.powerpc64-sony-ps3]
linker = "mold"
rustflags = [
    "-C", "relocation-model=static",
    "-C", "code-model=small",
    "-C", "target-feature=+altivec",
]
```

**Prerequisites:**

- A nightly Rust compiler with the `rust-src` component
- The [mold](https://github.com/rui314/mold) linker
- The [moldier](https://github.com/ZephyrCodesStuff/rust-ps3/tree/main/moldier) post-linker tool
- *(Optional)* `make_fself` or `scetool` for converting the output `.ELF` into an encrypted/signed `EBOOT.BIN` for running on real hardware.

**Build process:**

```bash
# Compile the binary
cargo +nightly build \
    --target powerpc64-sony-ps3 \
    -Z build-std=core,alloc \
    --release

# Patch the linked executable
moldier patch target/powerpc64-sony-ps3/release/my_program.ELF

# (Optional) Sign the binary for official hardware
make_fself "target/powerpc64-sony-ps3/release/my_program.ELF" "target/powerpc64-sony-ps3/release/my_program.BIN"
```

## Testing

The target fully supports running binaries (once they're patched), both on official hardware and on [open-source emulators](https://github.com/rpcs3/rpcs3).

As official firmware for the system forbids running unsigned code, the system must first be jailbroken in order to run binaries. This is not optional.

Emulators do not impose any requirement regarding codesigning, thus testing on emulators is straightforward.

Debugging is fully possible, either via debug firmware APIs on the official hardware, or on emulators via either their integrated debuggers, or a GDB server the emulator provides.

## Cross-compilation toolchains and C code

The target fully supports C/C++ code. Any compiler capable of producing binaries for a PowerPC64 big-endian processor can produce code to be embedded into the Rust program.
