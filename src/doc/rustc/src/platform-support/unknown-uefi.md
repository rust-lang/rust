# `*-unknown-uefi`

**Tier: 3**

Unified Extensible Firmware Interface (UEFI) targets for application, driver,
and core UEFI binaries.

Available targets:

- `aarch64-unknown-uefi`
- `i686-unknown-uefi`
- `x86_64-unknown-uefi`

## Target maintainers

- David Rheinsberg ([@dvdhrm](https://github.com/dvdhrm))
- Nicholas Bishop ([@nicholasbishop](https://github.com/nicholasbishop))

## Requirements

All UEFI targets can be used as `no-std` environments via cross-compilation.
Support for `std` is missing, but actively worked on. `alloc` is supported if
an allocator is provided by the user. No host tools are supported.

The UEFI environment resembles the environment for Microsoft Windows, with some
minor differences. Therefore, cross-compiling for UEFI works with the same
tools as cross-compiling for Windows. The target binaries are PE32+ encoded,
the calling convention is different for each architecture, but matches what
Windows uses (if the architecture is supported by Windows). The special
`efiapi` Rust calling-convention chooses the right ABI for the target platform
(`extern "C"` is incorrect on Intel targets at least). The specification has an
elaborate section on the different supported calling-conventions, if more
details are desired.

MMX, SSE, and other FP-units are disabled by default, to allow for compilation
of core UEFI code that runs before they are set up. This can be overridden for
individual compilations via rustc command-line flags. Not all firmwares
correctly configure those units, though, so careful inspection is required.

As native to PE32+, binaries are position-dependent, but can be relocated at
runtime if their desired location is unavailable. The code must be statically
linked. Dynamic linking is not supported. Code is shared via UEFI interfaces,
rather than dynamic linking. Additionally, UEFI forbids running code on
anything but the boot CPU/thread, nor is interrupt-usage allowed (apart from
the timer interrupt). Device drivers are required to use polling methods.

UEFI uses a single address-space to run all code in. Multiple applications can
be loaded simultaneously and are dispatched via cooperative multitasking on a
single stack.

By default, the UEFI targets use the `link`-flavor of the LLVM linker `lld` to
link binaries into the final PE32+ file suffixed with `*.efi`. The PE subsystem
is set to `EFI_APPLICATION`, but can be modified by passing `/subsystem:<...>`
to the linker. Similarly, the entry-point is to to `efi_main` but can be
changed via `/entry:<...>`. The panic-strategy is set to `abort`,

The UEFI specification is available online for free:
[UEFI Specification Directory](https://uefi.org/specifications)

## Building rust for UEFI targets

Rust can be built for the UEFI targets by enabling them in the `rustc` build
configuration. Note that you can only build the standard libraries. The
compiler and host tools currently cannot be compiled for UEFI targets. A sample
configuration would be:

```toml
[build]
build-stage = 1
target = ["x86_64-unknown-uefi"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building rust for UEFI targets" above), or build your own copy of `core` by
using `build-std`, `cargo-buildx`, or similar.

A native build with the unstable `build-std`-feature can be achieved via:

```sh
cargo +nightly build \
    -Zbuild-std=core,compiler_builtins \
    -Zbuild-std-features=compiler-builtins-mem \
    --target x86_64-unknown-uefi
```

Alternatively, you can install `cargo-xbuild` via
`cargo install --force cargo-xbuild` and build for the UEFI targets via:

```sh
cargo \
    +nightly \
    xbuild \
    --target x86_64-unknown-uefi
```

## Testing

UEFI applications can be copied into the ESP on any UEFI system and executed
via the firmware boot menu. The qemu suite allows emulating UEFI systems and
executing UEFI applications as well. See its documentation for details.

The [uefi-run](https://github.com/Richard-W/uefi-run) rust tool is a simple
wrapper around `qemu` that can spawn UEFI applications in qemu. You can install
it via `cargo install uefi-run` and execute qemu applications as
`uefi-run ./application.efi`.

## Cross-compilation toolchains and C code

There are 3 common ways to compile native C code for UEFI targets:

- Use the official SDK by Intel:
  [Tianocore/EDK2](https://github.com/tianocore/edk2). This supports a
  multitude of platforms, comes with the full specification transposed into C,
  lots of examples and build-system integrations. This is also the only
  officially supported platform by Intel, and is used by many major firmware
  implementations. Any code compiled via the SDK is compatible to rust binaries
  compiled for the UEFI targets. You can link them directly into your rust
  binaries, or call into each other via UEFI protocols.
- Use the **GNU-EFI** suite. This approach is used by many UEFI applications
  in the Linux/OSS ecosystem. The GCC compiler is used to compile ELF binaries,
  and linked with a pre-loader that converts the ELF binary to PE32+
  **at runtime**. You can combine such binaries with the rust UEFI targets only
  via UEFI protocols. Linking both into the same executable will fail, since
  one is an ELF executable, and one a PE32+. If linking to **GNU-EFI**
  executables is desired, you must compile your rust code natively for the same
  GNU target as **GNU-EFI** and use their pre-loader. This requires careful
  consideration about which calling-convention to use when calling into native
  UEFI protocols, or calling into linked **GNU-EFI** code (similar to how these
  differences need to be accounted for when writing **GNU-EFI** C code).
- Use native Windows targets. This means compiling your C code for the Windows
  platform as if it was the UEFI platform. This works for static libraries, but
  needs adjustments when linking into an UEFI executable. You can, however,
  link such static libraries seamlessly into rust code compiled for UEFI
  targets. Be wary of any includes that are not specifically suitable for UEFI
  targets (especially the C standard library includes are not always
  compatible). Freestanding compilations are recommended to avoid
  incompatibilites.

## Ecosystem

The rust language has a long history of supporting UEFI targets. Many crates
have been developed to provide access to UEFI protocols and make UEFI
programming more ergonomic in rust. The following list is a short overview (in
alphabetical ordering):

- **efi**: *Ergonomic Rust bindings for writing UEFI applications*. Provides
  _rustified_ access to UEFI protocols, implements allocators and a safe
  environment to write UEFI applications.
- **r-efi**: *UEFI Reference Specification Protocol Constants and Definitions*.
  A pure transpose of the UEFI specification into rust. This provides the raw
  definitions from the specification, without any extended helpers or
  _rustification_. It serves as baseline to implement any more elaborate rust
  UEFI layers.
- **uefi-rs**: *Safe and easy-to-use wrapper for building UEFI apps*. An
  elaborate library providing safe abstractions for UEFI protocols and
  features. It implements allocators and provides an execution environment to
  UEFI applications written in rust.
- **uefi-run**: *Run UEFI applications*. A small wrapper around _qemu_ to spawn
  UEFI applications in an emulated `x86_64` machine.

## Example: Freestanding

The following code is a valid UEFI application returning immediately upon
execution with an exit code of 0. A panic handler is provided. This is executed
by rust on panic. For simplicity, we simply end up in an infinite loop.

Note that as of rust-1.31.0, all features used here are stabilized. No unstable
features are required, nor do we rely on nightly compilers. However, if you do
not compile rustc for the UEFI targets, you need a nightly compiler to support
the `-Z build-std` flag.

This example can be compiled as binary crate via `cargo`:

```sh
cargo +nightly build \
    -Zbuild-std=core,compiler_builtins \
    -Zbuild-std-features=compiler-builtins-mem \
    --target x86_64-unknown-uefi
```

```rust,ignore (platform-specific,eh-personality-is-unstable)
#![no_main]
#![no_std]

#[panic_handler]
fn panic_handler(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[export_name = "efi_main"]
pub extern "C" fn main(_h: *mut core::ffi::c_void, _st: *mut core::ffi::c_void) -> usize {
    0
}
```

## Example: Hello World

This is an example UEFI application that prints "Hello World!", then waits for
key input before it exits. It serves as base example how to write UEFI
applications without any helper modules other than the standalone UEFI protocol
definitions provided by the `r-efi` crate.

This extends the "Freestanding" example and builds upon its setup. See there
for instruction how to compile this as binary crate.

Note that UEFI uses UTF-16 strings. Since rust literals are UTF-8, we have to
use an open-coded, zero-terminated, UTF-16 array as argument to
`output_string()`. Similarly to the panic handler, real applications should
rather use UTF-16 modules.

```rust,ignore (platform-specific,eh-personality-is-unstable)
#![no_main]
#![no_std]

use r_efi::efi;

#[panic_handler]
fn panic_handler(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[export_name = "efi_main"]
pub extern "C" fn main(_h: efi::Handle, st: *mut efi::SystemTable) -> efi::Status {
    let s = [
        0x0048u16, 0x0065u16, 0x006cu16, 0x006cu16, 0x006fu16, // "Hello"
        0x0020u16, //                                             " "
        0x0057u16, 0x006fu16, 0x0072u16, 0x006cu16, 0x0064u16, // "World"
        0x0021u16, //                                             "!"
        0x000au16, //                                             "\n"
        0x0000u16, //                                             NUL
    ];

    // Print "Hello World!".
    let r =
        unsafe { ((*(*st).con_out).output_string)((*st).con_out, s.as_ptr() as *mut efi::Char16) };
    if r.is_error() {
        return r;
    }

    // Wait for key input, by waiting on the `wait_for_key` event hook.
    let r = unsafe {
        let mut x: usize = 0;
        ((*(*st).boot_services).wait_for_event)(1, &mut (*(*st).con_in).wait_for_key, &mut x)
    };
    if r.is_error() {
        return r;
    }

    efi::Status::SUCCESS
}
```
