# `*-unknown-uefi`

**Tier: 2**

Unified Extensible Firmware Interface (UEFI) targets for application, driver,
and core UEFI binaries.

Available targets:

- `aarch64-unknown-uefi`
- `i686-unknown-uefi`
- `x86_64-unknown-uefi`

## Target maintainers

[@dvdhrm](https://github.com/dvdhrm)
[@nicholasbishop](https://github.com/nicholasbishop)

## Requirements

All UEFI targets can be used as `no-std` environments via cross-compilation.
Support for `std` is present, but incomplete and extremely new. `alloc` is supported if
an allocator is provided by the user or if using std. No host tools are supported.

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
to the linker. Similarly, the entry-point is set to `efi_main` but can be
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

Starting with Rust 1.67, precompiled artifacts are provided via
`rustup`. For example, to use `x86_64-unknown-uefi`:

```sh
# install cross-compile toolchain
rustup target add x86_64-unknown-uefi
# target flag may be used with any cargo or rustc command
cargo build --target x86_64-unknown-uefi
```

### Building a driver

There are three types of UEFI executables: application, boot service
driver, and runtime driver. All of Rust's UEFI targets default to
producing applications. To build a driver instead, pass a
[`subsystem`][linker-subsystem] linker flag with a value of
`efi_boot_service_driver` or `efi_runtime_driver`.

Example:

```toml
# In .cargo/config.toml:
[build]
rustflags = ["-C", "link-args=/subsystem:efi_runtime_driver"]
```

## Testing

UEFI applications can be copied into the ESP on any UEFI system and executed
via the firmware boot menu. The qemu suite allows emulating UEFI systems and
executing UEFI applications as well. See its documentation for details.

The [uefi-run] rust tool is a simple
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
  incompatibilities.

## Ecosystem

The rust language has a long history of supporting UEFI targets. Many crates
have been developed to provide access to UEFI protocols and make UEFI
programming more ergonomic in rust. The following list is a short overview (in
alphabetical ordering):

- **[efi][efi-crate]**: *Ergonomic Rust bindings for writing UEFI applications*. Provides
  _rustified_ access to UEFI protocols, implements allocators and a safe
  environment to write UEFI applications.
- **[r-efi]**: *UEFI Reference Specification Protocol Constants and Definitions*.
  A pure transpose of the UEFI specification into rust. This provides the raw
  definitions from the specification, without any extended helpers or
  _rustification_. It serves as baseline to implement any more elaborate rust
  UEFI layers.
- **[uefi-rs]**: *Safe and easy-to-use wrapper for building UEFI apps*. An
  elaborate library providing safe abstractions for UEFI protocols and
  features. It implements allocators and provides an execution environment to
  UEFI applications written in rust.
- **[uefi-run]**: *Run UEFI applications*. A small wrapper around _qemu_ to spawn
  UEFI applications in an emulated `x86_64` machine.

## Example: Freestanding

The following code is a valid UEFI application returning immediately upon
execution with an exit code of 0. A panic handler is provided. This is executed
by rust on panic. For simplicity, we simply end up in an infinite loop.

This example can be compiled as binary crate via `cargo`:

```sh
cargo build --target x86_64-unknown-uefi
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

## Rust std for UEFI
This section contains information on how to use std on UEFI.

### Build std
The building std part is pretty much the same as the official [docs](https://rustc-dev-guide.rust-lang.org/getting-started.html).
The linker that should be used is `rust-lld`. Here is a sample `bootstrap.toml`:
```toml
[rust]
lld = true
```
Then just build using `x.py`:
```sh
./x.py build --target x86_64-unknown-uefi --stage 1
```
Alternatively, it is possible to use the `build-std` feature. However, you must use a toolchain which has the UEFI std patches.
Then just build the project using the following command:
```sh
cargo build --target x86_64-unknown-uefi -Zbuild-std=std,panic_abort
```

### Implemented features
#### alloc
- Implemented using `EFI_BOOT_SERVICES.AllocatePool()` and `EFI_BOOT_SERVICES.FreePool()`.
- Passes all the tests.
- Currently uses `EfiLoaderData` as the `EFI_ALLOCATE_POOL->PoolType`.
#### cmath
- Provided by compiler-builtins.
#### env
- Just some global constants.
#### locks
- The provided locks should work on all standard single-threaded UEFI implementations.
#### os_str
- While the strings in UEFI should be valid UCS-2, in practice, many implementations just do not care and use UTF-16 strings.
- Thus, the current implementation supports full UTF-16 strings.
#### stdio
- Uses `Simple Text Input Protocol` and `Simple Text Output Protocol`.
- Note: UEFI uses CRLF for new line. This means Enter key is registered as CR instead of LF.
#### args
- Uses `EFI_LOADED_IMAGE_PROTOCOL->LoadOptions`

## Example: Hello World With std
The following code features a valid UEFI application, including `stdio` and `alloc` (`OsString` and `Vec`):

This example can be compiled as binary crate via `cargo` using the toolchain
compiled from the above source (named custom):

```sh
cargo +custom build --target x86_64-unknown-uefi
```

```rust,ignore (platform-specific)
#![feature(uefi_std)]

use r_efi::{efi, protocols::simple_text_output};
use std::{
  ffi::OsString,
  os::uefi::{env, ffi::OsStrExt}
};

pub fn main() {
  println!("Starting Rust Application...");

  // Use System Table Directly
  let st = env::system_table().as_ptr() as *mut efi::SystemTable;
  let mut s: Vec<u16> = OsString::from("Hello World!\n").encode_wide().collect();
  s.push(0);
  let r =
      unsafe {
        let con_out: *mut simple_text_output::Protocol = (*st).con_out;
        let output_string: extern "efiapi" fn(_: *mut simple_text_output::Protocol, *mut u16) -> efi::Status = (*con_out).output_string;
        output_string(con_out, s.as_ptr() as *mut efi::Char16)
      };
  assert!(!r.is_error())
}
```

### BootServices
The current implementation of std makes `BootServices` unavailable once `ExitBootServices` is called. Refer to [Runtime Drivers](https://edk2-docs.gitbook.io/edk-ii-uefi-driver-writer-s-guide/7_driver_entry_point/711_runtime_drivers) for more information regarding how to handle switching from using physical addresses to using virtual addresses.

Note: It should be noted that it is up to the user to drop all allocated memory before `ExitBootServices` is called.

[efi-crate]: https://github.com/gurry/efi
[linker-subsystem]: https://learn.microsoft.com/en-us/cpp/build/reference/subsystem
[r-efi]: https://github.com/r-efi/r-efi
[uefi-rs]: https://github.com/rust-osdev/uefi-rs
[uefi-run]: https://github.com/Richard-W/uefi-run
