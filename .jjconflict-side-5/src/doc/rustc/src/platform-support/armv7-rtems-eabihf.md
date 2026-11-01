# `armv7-rtems-eabihf`

**Tier: 3**

ARM targets for the [RTEMS realtime operating system](https://www.rtems.org)  using the RTEMS gcc cross-compiler for linking against the libraries of a specified Board Support Package (BSP).

## Target maintainers

[@thesummer](https://github.com/thesummer)

## Requirements

The target does not support host tools. Only cross-compilation is possible.
The cross-compiler toolchain can be obtained by following the installation instructions
of the [RTEMS Documentation](https://docs.rtems.org/docs/main/user/index.html). Additionally to the cross-compiler also a compiled BSP
for a board fitting the architecture needs to be available on the host.
Currently tested has been the BSP `xilinx_zynq_a9_qemu` of RTEMS 6.

`std` support is available, but not yet fully tested. Do NOT use in flight software!

The target follows the EABI calling convention for `extern "C"`.

The resulting binaries are in ELF format.

## Building the target

The target can be built by the standard compiler of Rust.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

In order to build an RTEMS executable it is also necessary to have a basic RTEMS configuration (in C) compiled to link against as this configures the operating system.
An example can be found at this [`rtems-sys`](https://github.com/thesummer/rtems-sys) crate which could be added as an dependency to your application.

## Testing

The resulting binaries run fine on an emulated target (possibly also on a real Zedboard or similar).
For example, on qemu the following command can execute the binary:
```sh
qemu-system-arm -no-reboot -serial null -serial mon:stdio -net none -nographic -M xilinx-zynq-a9 -m 512M -kernel <binary file>
```

While basic execution of the unit test harness seems to work. However, running the Rust testsuite on the (emulated) hardware has not yet been tested.

## Cross-compilation toolchains and C code

Compatible C-code can be built with the RTEMS cross-compiler toolchain `arm-rtems6-gcc`.
For more information how to build the toolchain, RTEMS itself and RTEMS applications please have a look at the [RTEMS Documentation](https://docs.rtems.org/docs/main/user/index.html).
