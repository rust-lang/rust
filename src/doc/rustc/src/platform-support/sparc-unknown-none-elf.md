# `sparc-unknown-none-elf`

**Tier: 3**

Rust for bare-metal 32-bit SPARC V7 and V8 systems, e.g. the Gaisler LEON3.

| Target                 | Descriptions                              |
| ---------------------- | ----------------------------------------- |
| sparc-unknown-none-elf | SPARC V7 32-bit (freestanding, hardfloat) |

## Target maintainers

- Jonathan Pallant, <jonathan.pallant@ferrous-systems.com>, https://ferrous-systems.com

## Requirements

This target is cross-compiled. There is no support for `std`. There is no
default allocator, but it's possible to use `alloc` by supplying an allocator.

By default, code generated with this target should run on any `SPARC` hardware;
enabling additional target features may raise this baseline.

- `-Ctarget-cpu=v8` adds the extra SPARC V8 instructions.

- `-Ctarget-cpu=leon3` adds the SPARC V8 instructions and sets up scheduling to
  suit the Gaisler Leon3.

Functions marked `extern "C"` use the [standard SPARC architecture calling
convention](https://sparc.org/technical-documents/).

This target generates ELF binaries. Any alternate formats or special
considerations for binary layout will require linker options or linker scripts.

## Building the target

You can build Rust with support for the target by adding it to the `target`
list in `config.toml`:

```toml
[build]
build-stage = 1
host = ["<target for your host>"]
target = ["<target for your host>", "sparc-unknown-none-elf"]
```

Replace `<target for your host>` with `x86_64-unknown-linux-gnu` or whatever
else is appropriate for your host machine.

## Building Rust programs

To build with this target, pass it to the `--target` argument, like:

```console
cargo build --target sparc-unknown-none-elf
```

This target uses GCC as a linker, and so you will need an appropriate GCC
compatible `sparc-unknown-none` toolchain. The default linker binary is
`sparc-elf-gcc`, but you can override this in your project configuration, as
follows:

`.cargo/config.toml`:
```toml
[target.sparc-unknown-none-elf]
linker = "sparc-custom-elf-gcc"
```

## Testing

As `sparc-unknown-none-elf` supports a variety of different environments and does
not support `std`, this target does not support running the Rust test suite.

## Cross-compilation toolchains and C code

This target was initially tested using [BCC2] from Gaisler, along with the TSIM
Leon3 processor simulator. Both [BCC2] GCC and [BCC2] Clang have been shown to
work. To work with these tools, your project configuration should contain
something like:

[BCC2]: https://www.gaisler.com/index.php/downloads/compilers

`.cargo/config.toml`:
```toml
[target.sparc-unknown-none-elf]
linker = "sparc-gaisler-elf-gcc"
runner = "tsim-leon3"

[build]
target = ["sparc-unknown-none-elf"]
rustflags = "-Ctarget-cpu=leon3"
```

With this configuration, running `cargo run` will compile your code for the
SPARC V8 compatible Gaisler Leon3 processor and then start the `tsim-leon3`
simulator. The `libcore` was pre-compiled as part of the `rustc` compilation
process using the SPARC V7 baseline, but if you are using a nightly toolchain
you can use the
[`-Z build-std=core`](https://doc.rust-lang.org/cargo/reference/unstable.html#build-std)
option to rebuild `libcore` from source. This may be useful if you want to
compile it for SPARC V8 and take advantage of the extra instructions.

`.cargo/config.toml`:
```toml
[target.sparc-unknown-none-elf]
linker = "sparc-gaisler-elf-gcc"
runner = "tsim-leon3"

[build]
target = ["sparc-unknown-none-elf"]
rustflags = "-Ctarget-cpu=leon3"

[unstable]
build-std = ["core"]
```

Either way, once the simulator is running, simply enter the command `run` to
start the code executing in the simulator.

The default C toolchain libraries are linked in, so with the Gaisler [BCC2]
toolchain, and using its default Leon3 BSP, you can use call the C `putchar`
function and friends to output to the simulator console. The default linker
script is also appropriate for the Leon3 simulator, so no linker script is
required.

Here's a complete example using the above config file:

```rust,ignore (cannot-test-this-because-it-assumes-special-libc-functions)
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

```console
$ cargo run --target=sparc-unknown-none-elf
   Compiling sparc-demo-rust v0.1.0 (/work/sparc-demo-rust)
    Finished dev [unoptimized + debuginfo] target(s) in 3.44s
     Running `tsim-leon3 target/sparc-unknown-none-elf/debug/sparc-demo-rust`

 TSIM3 LEON3 SPARC simulator, version 3.1.9 (evaluation version)

 Copyright (C) 2023, Frontgrade Gaisler - all rights reserved.
 This software may only be used with a valid license.
 For latest updates, go to https://www.gaisler.com/
 Comments or bug-reports to support@gaisler.com

 This TSIM evaluation version will expire 2023-11-28

Number of CPUs: 2
system frequency: 50.000 MHz
icache: 1 * 4 KiB, 16 bytes/line (4 KiB total)
dcache: 1 * 4 KiB, 16 bytes/line (4 KiB total)
Allocated 8192 KiB SRAM memory, in 1 bank at 0x40000000
Allocated 32 MiB SDRAM memory, in 1 bank at 0x60000000
Allocated 8192 KiB ROM memory at 0x00000000
section: .text, addr: 0x40000000, size: 20528 bytes
section: .rodata, addr: 0x40005030, size: 128 bytes
section: .data, addr: 0x400050b0, size: 1176 bytes
read 347 symbols

tsim> run
  Initializing and starting from 0x40000000
Hello, this is Rust!

  Program exited normally on CPU 0.
tsim>
```
