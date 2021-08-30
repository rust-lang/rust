# m68k-unknown-linux-gnu

**Tier: 3**

Motorola 680x0 Linux

## Designated Developers

* [@glaubitz](https://github.com/glaubitz)
* [@ricky26](https://github.com/ricky26)

## Requirements

This target requires a Linux/m68k build environment for cross-compilation which
is available on Debian and Debian-based systems, openSUSE and other distributions.

On Debian, it should be sufficient to install a g++ cross-compiler for the m68k
architecture which will automatically pull in additional dependencies such as
the glibc cross development package:

```text
# apt install g++-m68k-linux-gnu
```

Binaries can be run using QEMU user emulation. On Debian-based systems, it should be
sufficient to install the package `qemu-user-static` to be able to run simple static
binaries:

```text
# apt install qemu-user-static
```

To run more complex programs, it will be necessary to set up a Debian/m68k chroot with
the help of the command `debootstrap`:

```text
# apt install debootstrap debian-ports-archive-keyring
# debootstrap --keyring=/usr/share/keyrings/debian-ports-archive-keyring.gpg --arch=m68k unstable debian-68k http://ftp.ports.debian.org/debian-ports
```

This chroot can then seamlessly entered using the normal `chroot` command thanks to
QEMU user emulation:

```text
# chroot /path/to/debian-68k
```

To get started with native builds, which are currently untested, a native Debian/m68k
system can be installed either on real hardware such as 68k-based Commodore Amiga or
Atari systems or emulated environments such as QEMU version 4.2 or newer or ARAnyM.

ISO images for installation are provided by the Debian Ports team and can be obtained
from the Debian CD image server available at:

[https://cdimage.debian.org/cdimage/ports/current](https://cdimage.debian.org/cdimage/ports/current/)

Documentation for Debian/m68k is available on the Debian Wiki at:

[https://wiki.debian.org/M68k](https://wiki.debian.org/M68k)

Support is available either through the `debian-68k` mailing list:

[https://lists.debian.org/debian-68k/](https://lists.debian.org/debian-68k/)

or the `#debian-68k` IRC channel on OFTC network.

## Building

The codegen for this target should be built by default. However, core and std
are currently missing but are being worked on and should become available in
the near future.

## Cross-compilation

This target can be cross-compiled from a standard Debian or Debian-based, openSUSE or any
other distribution which has a basic m68k cross-toolchain available.

## Testing

Currently there is no support to run the rustc test suite for this target.

## Building Rust programs

Rust programs can be built for that target:

```text
rustc --target m68k-unknown-linux-gnu your-code.rs
```

Very simple progams can be run using the `qemu-m68k-static` program:

```text
$ qemu-m68k-static your-code
```

For more complex applications, a chroot or native (emulated) Debian/m68k system are required
for testing.
