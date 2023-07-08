# \*-unknown-netbsd

**Tier: 3**

[NetBSD] multi-platform 4.4BSD-based UNIX-like operating system.

[NetBSD]: https://www.NetBSD.org/

The target names follow this format: `$ARCH-unknown-netbsd{-$SUFFIX}`,
where `$ARCH` specifies the target processor architecture and
`-$SUFFIX` (optional) might indicate the ABI. The following targets
are currently defined running NetBSD:

|          Target name           | NetBSD Platform |
|--------------------------------|-----------------|
| `amd64-unknown-netbsd`         | [amd64 / x86_64 systems](https://wiki.netbsd.org/ports/amd64/) |
| `armv7-unknown-netbsd-eabihf`  | [32-bit ARMv7 systems with hard-float](https://wiki.netbsd.org/ports/evbarm/) |
| `armv6-unknown-netbsd-eabihf`  | [32-bit ARMv6 systems with hard-float](https://wiki.netbsd.org/ports/evbarm/) |
| `aarch64-unknown-netbsd`       | [64-bit ARM systems, little-endian](https://wiki.netbsd.org/ports/evbarm/) |
| `aarch64_be-unknown-netbsd`    | [64-bit ARM systems, big-endian](https://wiki.netbsd.org/ports/evbarm/) |
| `i586-unknown-netbsd`          | [32-bit i386, restricted to Pentium](https://wiki.netbsd.org/ports/i386/) |
| `i686-unknown-netbsd`          | [32-bit i386 with SSE](https://wiki.netbsd.org/ports/i386/) |
| `mipsel-unknown-netbsd`        | [32-bit mips, requires mips32 cpu support](https://wiki.netbsd.org/ports/evbmips/) |
| `powerpc-unknown-netbsd`       | [Various 32-bit PowerPC systems, e.g. MacPPC](https://wiki.netbsd.org/ports/macppc/) |
| `riscv64gc-unknown-netbsd`     | [64-bit RISC-V](https://wiki.netbsd.org/ports/riscv/)
| `sparc64-unknown-netbsd`       | [Sun UltraSPARC systems](https://wiki.netbsd.org/ports/sparc64/) |

All use the "native" `stdc++` library which goes along with the natively
supplied GNU C++ compiler for the given OS version.  Many of the bootstraps
are built for NetBSD 9.x, although some exceptions exist (some
are built for NetBSD 8.x but also work on newer OS versions).


## Designated Developers

- [@he32](https://github.com/he32), `he@NetBSD.org`
- [NetBSD/pkgsrc-wip's rust](https://github.com/NetBSD/pkgsrc-wip/blob/master/rust/Makefile) maintainer (see MAINTAINER variable). This package is part of "pkgsrc work-in-progress" and is used for deployment and testing of new versions of rust
- [NetBSD's pkgsrc lang/rust](https://github.com/NetBSD/pkgsrc/tree/trunk/lang/rust) for the "proper" package in pkgsrc.
- [NetBSD's pkgsrc lang/rust-bin](https://github.com/NetBSD/pkgsrc/tree/trunk/lang/rust-bin) which re-uses the bootstrap kit as a binary distribution and therefore avoids the rather protracted native build time of rust itself

Fallback to pkgsrc-users@NetBSD.org, or fault reporting via NetBSD's
bug reporting system.

## Requirements

The `amd64-unknown-netbsd` artifacts is being distributed by the
rust project.

The other targets are built by the designated developers (see above),
and the targets are initially cross-compiled, but many if not most
of them are also built natively as part of testing.


## Building

The default build mode for the packages is a native build.


## Cross-compilation

These targets can be cross-compiled, and we do that via the pkgsrc
package(s).

Cross-compilation typically requires the "tools" and "dest" trees
resulting from a normal cross-build of NetBSD itself, ref. our main
build script, `build.sh`.

See e.g. [do-cross.mk
Makefile](https://github.com/NetBSD/pkgsrc/tree/trunk/lang/rust/do-cross.mk)
for the Makefile used to cross-build all the above NetBSD targets
(except for the `amd64` target).

The major option for the rust build is whether to build rust with
the LLVM rust carries in its distribution, or use the LLVM package
installed from pkgsrc.  The `PKG_OPTIONS.rust` option is
`rust-internal-llvm`, ref.  [the rust package's options.mk make
fragment](https://github.com/NetBSD/pkgsrc/blob/trunk/lang/rust/options.mk).
It defaults to being set for a few of the above platforms, for
various reasons (see comments), but is otherwise unset and therefore
indicates use of the pkgsrc LLVM.


## Testing

The Rust testsuite could presumably be run natively.

For the systems where the maintainer can build natively, the rust
compiler itself is re-built natively.  This involves the rust compiler
being re-built with the newly self-built rust compiler, so excercises
the result quite extensively.

Additionally, for some systems we build `librsvg`, and for the more
capable systems we build and test `firefox` (amd64, i386, aarch64).


## Building Rust programs

Rust ships pre-compiled artifacts for the `amd64-unknown-netbsd`
target.

For the other systems mentioned above, using the `pkgsrc` route is
probably the easiest, possibly via the `rust-bin` package to save
time, see the `RUST_TYPE` variable from the `rust.mk` Makefile
fragment.

The pkgsrc rust package has a few files to assist with building
pkgsrc packages written in rust, ref. the `rust.mk` and `cargo.mk`
Makefile fragments in the `lang/rust` package.
