# `bpf*-unknown-none`

**Tier: 3**

* `bpfeb-unknown-none` (big endian)
* `bpfel-unknown-none` (little endian)

Targets for the 64-bit [BPF virtual machine][ebpf].

## Target maintainers

- [@alessandrod](https://github.com/alessandrod)
- [@dave-tucker](https://github.com/dave-tucker)
- [@tamird](https://github.com/tamird)
- [@vadorovsky](https://github.com/vadorovsky)

## Requirements

BPF targets require a Rust toolchain with the `rust-src` component. In
addition, you must install the [bpf-linker].

They don't support std and alloc and are meant for a `no_std` environment.

`extern "C"` uses the [BPF ABI calling convention][bpf-abi].

Produced binaries use the ELF format.

## Building the target

You can build Rust with support for BPF targets by adding them to the `target`
list in `config.toml`:

```toml
[build]
target = ["bpfeb-unknown-none", "bpfel-unknown-none"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

Building the BPF target requires specifying it explicitly. Users can either
add it to the `target` list in `config.toml`:

```toml
[build]
target = ["bpfel-unknown-none"]
```

Or specify it directly in the `cargo build` invocation:

```console
cargo +nightly build -Z build-std=core --target bpfel-unknown-none
```

BPF has its own debug info format called [BTF][btf].

BPF targets use [bpf-linker], an LLVM bitcode linker, which by
default strips the debug info, but it has an experimental feature of emitting
BTF, which can be enabled by adding `-C link-arg=--btf` to `RUSTFLAGS`. With
that feature enabled, [bpf-linker] does not only link different
crates/modules, but also performs a necessary sanitization of debug info, which
is required to produce valid [BTF][btf] acceptable by the Linux kernel.

## Error handling

There is no concept of stack unwinding in BPF, therefore BPF programs are
expected to handle errors in a recoverable manner. Therefore most BPF programs
written in Rust use the following no-op panic handler implementation:

```rust,ignore
#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
```

Infinite loops are forbidden by the BPF verifier. Therefore, if the program
contains any code which can panic, the BPF VM refuses to load it.

## Testing

BPF bytecode needs to be executed on a BPF virtual machine, like the one
provided by the Linux kernel or one of the user-space implementations like
[rbpf][rbpf]. None of them support running Rust testsuite. One of the reasons
is the lack of support for panicking.

Therefore, unit tests need to run on the host system. That requirement can be
enforced by the following conditional check:

```rust,ignore
#[cfg(all(not(target_arch = "bpf"), test))]
mod test {}
```

## Cross-compilation toolchains

BPF programs are always cross-compiled from a host (e.g.
`x86_64-unknown-linux-*`) for a BPF target (e.g. `bpfel-unknown-none`).

The endianness of a chosen BPF target needs to match the endianness of the BPF
VM host on which the program is supposed to run.

The architecture of the BPF VM host often has an impact on types that the BPF
programs should use. For example [kprobes][kprobe], [fprobes][fprobe] and
[uprobes][uprobe] allow dynamic function tracing and lookup into host registers
through the [`pt_regs`][pt-regs] struct, which differs across architectures.

That difference is still not a concern of the compiler. Instead, it should be
handled by the developers. [Aya][aya] (the library for writing Linux BPF
programs and the main consumer of BPF targets in Rust) handles that by
providing the [`aya-ebpf-cty`][aya-ebpf-cty] crate, with type aliases similar
to those provided by [`core:ffi`][core-ffi]. [`aya-ebpf-cty`][aya-ebpf-cty]
allows to specify the VM target through the `CARGO_CFG_BPF_TARGET_ARCH`
environment variable (e.g. `CARGO_CFG_BPF_TARGET_ARCH=aarch64`).

## C code

It's possible to link a Rust BPF project to bitcode or object files which are
built from C code with [clang][clang]. It can be done using a `rustc-link-lib`
instruction in `build.rs`. Example:

```rust
use std::{env, process::Command};

let out_dir = env::var("OUT_DIR").unwrap();
let c_module = "my_module.bpf.c";
let s = Command::new("clang")
    .arg("-I")
    .arg("src/")
    .arg("-O2")
    .arg("-emit-llvm")
    .arg("-target")
    .arg("bpf")
    .arg("-c")
    .arg("-g")
    .arg(c_module)
    .arg("-o")
    .arg(format!("{out_dir}/my_module.bpf.o"))
    .status()
    .unwrap();
assert!(s.success());
println!("cargo:rustc-link-search=native={out_dir}");
println!("cargo:rustc-link-lib=link-arg={out_dir}/my_module.bpf.o");
```

[ebpf]: https://ebpf.io/
[bpf-linker]: https://github.com/aya-rs/bpf-linker
[bpf-abi]: https://www.kernel.org/doc/html/v6.13-rc5/bpf/standardization/abi.html
[btf]: https://www.kernel.org/doc/html/latest/bpf/btf.html
[rbpf]: https://github.com/qmonnet/rbpf
[kprobe]: https://www.kernel.org/doc/html/latest/trace/kprobes.html
[fprobe]: https://www.kernel.org/doc/html/latest/trace/fprobe.html
[uprobe]: https://www.kernel.org/doc/html/latest/trace/uprobetracer.html
[pt-regs]: https://elixir.bootlin.com/linux/v6.12.6/source/arch/x86/include/uapi/asm/ptrace.h#L44
[aya]: https://aya-rs.dev
[aya-ebpf-cty]: https://github.com/aya-rs/aya/tree/main/ebpf/aya-ebpf-cty
[core-ffi]: https://doc.rust-lang.org/stable/core/ffi/index.html
[clang]: https://clang.llvm.org/
