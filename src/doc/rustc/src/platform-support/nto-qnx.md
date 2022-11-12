# nto-qnx

**Tier: 3**

[BlackBerry® QNX®][BlackBerry] Neutrino (nto) Real-time operating system.
The support has been implemented jointly by [Elektrobit Automotive GmbH][Elektrobit]
and [BlackBerry][BlackBerry].

[BlackBerry]: https://blackberry.qnx.com
[Elektrobit]: https://www.elektrobit.com

## Target maintainers

- Florian Bartels, `Florian.Bartels@elektrobit.com`, https://github.com/flba-eb
- Tristan Roach, `TRoach@blackberry.com`, https://github.com/gh-tr

## Requirements

Currently, only cross-compilation for QNX Neutrino on AArch64 and x86_64 are supported (little endian).
Adding other architectures that are supported by QNX Neutrino is possible.

The standard library does not yet support QNX Neutrino. Therefore, only `no_std` code can
be compiled.

`core` and `alloc` (with default allocator) are supported.

Applications must link against `libc.so` (see example). This is required because applications
always link against the `crt` library and `crt` depends on `libc.so`.

The correct version of `qcc` must be available by setting the `$PATH` variable (e.g. by sourcing `qnxsdp-env.sh` of the
QNX Neutrino toolchain).

### Small example application

```rust,ignore (platform-specific)
#![no_std]
#![no_main]
#![feature(lang_items)]

// We must always link against libc, even if no external functions are used
// "extern C" - Block can be empty but must be present
#[link(name = "c")]
extern "C" {
    pub fn printf(format: *const core::ffi::c_char, ...) -> core::ffi::c_int;
}

#[no_mangle]
pub extern "C" fn main(_argc: isize, _argv: *const *const u8) -> isize {
    const HELLO: &'static str = "Hello World, the answer is %d\n\0";
    unsafe {
        printf(HELLO.as_ptr() as *const _, 42);
    }
    0
}

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_panic: &PanicInfo<'_>) -> ! {
    loop {}
}

#[lang = "eh_personality"]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
```

The QNX Neutrino support of Rust has been tested with QNX Neutrino 7.1.

There are no further known requirements.

## Conditional compilation

For conditional compilation, following QNX Neutrino specific attributes are defined:

- `target_os` = `"nto"`
- `target_env` = `"nto71"` (for QNX Neutrino 7.1)

## Building the target

1. Create a `config.toml`

Example content:

```toml
profile = "compiler"
changelog-seen = 2
```

2. Compile the Rust toolchain for an `x86_64-unknown-linux-gnu` host (for both `aarch64` and `x86_64` targets)

Run the following:

```bash
env \
    CC_aarch64-unknown-nto-qnx7.1.0="qcc" \
    CFLAGS_aarch64-unknown-nto-qnx7.1.0="-Vgcc_ntoaarch64le_cxx" \
    CXX_aarch64-unknown-nto-qnx7.1.0="qcc" \
    AR_aarch64_unknown_nto_qnx7.1.0="ntoaarch64-ar" \
    CC_x86_64-pc-nto-qnx7.1.0="qcc" \
    CFLAGS_x86_64-pc-nto-qnx7.1.0="-Vgcc_ntox86_64_cxx" \
    CXX_x86_64-pc-nto-qnx7.1.0="qcc" \
    AR_x86_64_pc_nto_qnx7.1.0="ntox86_64-ar" \
        ./x.py build --target aarch64-unknown-nto-qnx7.1.0 --target x86_64-pc-nto-qnx7.1.0 --target x86_64-unknown-linux-gnu rustc library/core library/alloc/
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for this target, you must either build Rust with the target enabled (see "Building the target" above), or build your own copy of  `core` by using
`build-std` or similar.

## Testing

Compiled executables can directly be run on QNX Neutrino.

## Cross-compilation toolchains and C code

Compiling C code requires the same environment variables to be set as compiling the Rust toolchain (see above), to ensure `qcc` is used with proper arguments. To ensure compatibility, do not specify any further arguments that for example change calling conventions or memory layout.
