# `hexagon-unknown-none-elf`

**Tier: 3**

Rust for baremetal Hexagon DSPs.

| Target                   | Descriptions                              |
| ------------------------ | ----------------------------------------- |
| hexagon-unknown-none-elf | Hexagon 32-bit (freestanding, hardfloat)  |

## Target maintainers

- [Brian Cain](https://github.com/androm3da), `bcain@quicinc.com`

## Requirements

This target is cross-compiled.  There is no support for `std`. There is no
default allocator, but it's possible to use `alloc` by supplying an allocator.

By default, code generated with this target should run on Hexagon DSP hardware.

- `-Ctarget-cpu=hexagonv73` adds support for instructions defined up to Hexagon V73.

Functions marked `extern "C"` use the [Hexagon architecture calling convention](https://lists.llvm.org/pipermail/llvm-dev/attachments/20190916/21516a52/attachment-0001.pdf).

This target generates PIC ELF binaries.

## Building the target

You can build Rust with support for the target by adding it to the `target`
list in `bootstrap.toml`:

```toml
[build]
build-stage = 1
host = ["<target for your host>"]
target = ["<target for your host>", "hexagon-unknown-none-elf"]

[target.hexagon-unknown-none-elf]

cc = "hexagon-unknown-none-elf-clang"
cxx = "hexagon-unknown-none-elf-clang++"
linker = "hexagon-unknown-none-elf-clang"
ranlib = "hexagon-unknown-none-elf-ranlib"
ar = "hexagon-unknown-none-elf-ar"
llvm-libunwind = 'in-tree'
```

Replace `<target for your host>` with `x86_64-unknown-linux-gnu` or whatever
else is appropriate for your host machine.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Since `hexagon-unknown-none-elf` supports a variety of different environments and
does not support `std`, this target does not support running the Rust test suite.

## Cross-compilation toolchains and C code

This target has been tested using `qemu-system-hexagon`.

A common use case for `hexagon-unknown-none-elf` is building libraries that
link against C code and can be used in emulation or on a device with a
Hexagon DSP.

The Hexagon SDK has libraries which are useful to link against when running
on a device.


# Standalone OS

The script below will build an executable against "hexagon standalone OS"
which is suitable for emulation or bare-metal on-device testing.

First, run `cargo new --bin demo1_hexagon` then add the source below as
`src/main.rs`.  This program demonstrates the console output via semihosting.

```rust,ignore (platform-specific,eh-personality-is-unstable)
#![no_std]
#![no_main]

extern "C" {
    fn putchar(ch: i32);
    fn _exit(code: i32) -> !;
}

#[no_mangle]
extern "C" fn main() -> i32 {
    let message = "Hello, this is Rust!";
    for b in message.bytes() {
        unsafe {
            putchar(b as i32);
        }
    }
    0
}

#[panic_handler]
fn panic(_panic: &core::panic::PanicInfo) -> ! {
    unsafe {
        _exit(1);
    }
}

```

Next, save the script below as `build.sh` and edit it to suit your
environment.

* `hex_toolchain` below refers to the [hexagon toolchain using exclusively
public open source repos](https://github.com/quic/toolchain_for_hexagon/releases).
* `cc` below refers to clang.  You can use `clang` from your distribution, as
long as it's at least `clang-17`.  Or you can use
`hexagon-unknown-none-elf-clang` from one of the [hexagon open source toolchain
releases](https://github.com/quic/toolchain_for_hexagon/releases).

```sh
# Hexagon SDK, required for target libraries:
hex_sdk_root=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.3.0.0
hex_sdk_toolchain=${hex_sdk_root}/tools/HEXAGON_Tools/8.6.06

sdk_libs=${hex_sdk_toolchain}/Tools/target/hexagon/lib
q6_arch=v65
g0_lib_path=${sdk_libs}/${q6_arch}/G0
pic_lib_path=${sdk_libs}/${q6_arch}/G0/pic

build_cfg=release
cargo build --target=hexagon-unknown-none-elf -Zbuild-std --release

# Builds an executable against "hexagon standalone OS" suitable for emulation:
${cc} --target=hexagon-unknown-none-elf -o testit \
    -fuse-ld=lld \
    -m${q6_arch} \
    -nodefaultlibs \
    -nostartfiles \
    ${g0_lib_path}/crt0_standalone.o \
    ${g0_lib_path}/crt0.o \
    ${g0_lib_path}/init.o \
    -L${sdk_libs}/${q6_arch}/ \
    -L${sdk_libs}/ \
    wrap.c \
    target/hexagon-unknown-none-elf/${build_cfg}/libdemo1_hexagon.rlib \
    target/hexagon-unknown-none-elf/${build_cfg}/deps/libcore-*.rlib \
    target/hexagon-unknown-none-elf/${build_cfg}/deps/libcompiler_builtins-*.rlib \
    -Wl,--start-group \
    -Wl,--defsym,_SDA_BASE_=0,--defsym,__sbss_start=0,--defsym,__sbss_end=0 \
    ${g0_lib_path}/libstandalone.a \
    ${g0_lib_path}/libc.a \
    -lgcc \
    -lc_eh \
    -Wl,--end-group \
    ${g0_lib_path}/fini.o \

${hex_toolchain}/x86_64-linux-gnu/bin/qemu-system-hexagon -monitor none -display none -kernel ./testit
```

# QuRT OS

First, run `cargo new --lib demo2_hexagon` then add the source below as
`src/lib.rs`.  This program demonstrates inline assembly and console output
via semihosting.

```rust,ignore (platform-specific,eh-personality-is-unstable)
#![no_std]
#![no_main]
#![feature(lang_items)]
#![feature(asm_experimental_arch)]

use core::arch::asm;

extern "C" {
    fn putchar(ch: i32);
    fn _exit(code: i32) -> !;
}

fn hexagon_specific() {
    let mut buffer = [0_u8; 128];

    unsafe {
        let mut x = &buffer;
        asm!(
                "{{\n\t",
                "  v0=vmem({addr}+#0)\n\t",
                "  {tmp} = and({tmp}, #1)\n\t",
                "}}\n\t",
                addr = in(reg) x,
                tmp = out(reg) _,
            );
    }
}

#[no_mangle]
extern "C" fn hello() -> i32 {
    let message = "Hello, this is Rust!\n";
    for b in message.bytes() {
        unsafe {
            putchar(b as i32);
        }
    }
    hexagon_specific();
    0
}

#[panic_handler]
fn panic(_panic: &core::panic::PanicInfo) -> ! {
    unsafe {
        _exit(1);
    }
}

#[lang = "eh_personality"]
fn rust_eh_personality() {}

```

Next, create a C program as an entry point, save the content below as
`wrap.c`:

```C
int hello();

int main() {
    hello();
}
```

Then, save the script below as `build.sh` and edit it to suit your
environment.  The script below will build a shared object against the QuRT
RTOS which is suitable for emulation or on-device testing when loaded via
the fastrpc-shell.


```sh
# Hexagon SDK, required for target libraries:
hex_sdk_root=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.3.0.0
hex_sdk_toolchain=${hex_sdk_root}/tools/HEXAGON_Tools/8.6.06

sdk_libs=${hex_sdk_toolchain}/Tools/target/hexagon/lib
q6_arch=v65
g0_lib_path=${sdk_libs}/${q6_arch}/G0
pic_lib_path=${sdk_libs}/${q6_arch}/G0/pic
runelf=${hex_sdk_root}/rtos/qurt/computev65/sdksim_bin/runelf.pbn
rmohs=${hex_sdk_root}/libs/run_main_on_hexagon/ship/hexagon_toolv86_${q6_arch}/run_main_on_hexagon_sim

# Builds a library suitable for loading into "run_main_on_hexagon_sim" for
# emulation or frpc shell on real target:
${cc} --target=hexagon-unknown-none-elf -o testit.so \
    -fuse-ld=lld \
    -fPIC -shared \
    -nostdlib \
    -Wl,-Bsymbolic \
      -Wl,--wrap=malloc \
      -Wl,--wrap=calloc \
      -Wl,--wrap=free \
      -Wl,--wrap=realloc \
      -Wl,--wrap=memalign \
    -m${q6_arch} \
    wrap.c \
    target/hexagon-unknown-none-elf/${build_cfg}/libdemo2_hexagon.rlib \
    target/hexagon-unknown-none-elf/${build_cfg}/deps/libcore-*.rlib \
    target/hexagon-unknown-none-elf/${build_cfg}/deps/libcompiler_builtins-*.rlib \
    -Wl,-soname=testit \
    ${pic_lib_path}/libc.so

# -Bsymbolic above for memory alloc funcs is necessary to access the heap on
# target, but otherwise not required.

# multi-stage loader: runelf => run_main_on_hexagon_sim => testit.so{`main`}
${hex_toolchain}/x86_64-linux-gnu/bin/qemu-system-hexagon \
    -monitor none \
    -display none \
    -kernel ${runelf} \
    -append "${rmohs} -- ./testit.so"
```
