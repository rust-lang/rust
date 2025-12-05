# `avr-none`

Series of microcontrollers from Atmel: ATmega8, ATmega328p etc.

**Tier: 3**

## Target maintainers

[@Patryk27](https://github.com/Patryk27)

## Requirements

This target is only cross-compiled; x86-64 Linux, x86-64 macOS and aarch64 macOS
hosts are confirmed to work, but in principle any machine able to run rustc and
avr-gcc should be good.

Compiling for this target requires `avr-gcc` installed, because a couple of
intrinsics (like 32-bit multiplication) rely on [`libgcc`](https://github.com/gcc-mirror/gcc/blob/3269a722b7a03613e9c4e2862bc5088c4a17cc11/libgcc/config/avr/lib1funcs.S)
and can't be provided through `compiler-builtins` yet. This is a limitation that
[we hope to lift in the future](https://github.com/rust-lang/compiler-builtins/issues/711).

You'll also need to setup the `.cargo/config` file - see below for details.

## Building the target

Rust comes with AVR support enabled, you don't have to rebuild the compiler.

## Building Rust programs

Install `avr-gcc`:

```console
# Ubuntu:
$ sudo apt-get install gcc-avr

# Mac:
$ brew tap osx-cross/avr && brew install avr-gcc

# NixOS (takes a couple of minutes, since it's not provided through Hydra):
$ nix shell nixpkgs#pkgsCross.avr.buildPackages.gcc11
```

... setup `.cargo/config` for your project:

```toml
[build]
target = "avr-none"
rustflags = ["-C", "target-cpu=atmega328p"]

[unstable]
build-std = ["core"]
```

... and then simply run:

```console
$ cargo build --release
```

The final binary will be placed into
`./target/avr-none/release/your-project.elf`.

Note that since AVRs have rather small amounts of registers, ROM and RAM, it's
recommended to always use `--release` to avoid running out of space.

Also, please note that specifying `-C target-cpu` is required - here's a list of
the possible variants:

https://github.com/llvm/llvm-project/blob/093d4db2f3c874d4683fb01194b00dbb20e5c713/clang/lib/Basic/Targets/AVR.cpp#L32

## Testing

You can use [`simavr`](https://github.com/buserror/simavr) to emulate the
resulting firmware on your machine:

```console
$ simavr -m atmega328p ./target/avr-none/release/your-project.elf
```

Alternatively, if you want to write a couple of actual `#[test]`s, you can use
[`avr-tester`](https://github.com/Patryk27/avr-tester).
