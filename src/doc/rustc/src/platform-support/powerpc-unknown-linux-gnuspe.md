# powerpc-unknown-linux-gnuspe

**Tier: 3**

`powerpc-unknown-linux-gnuspe` is a target for Linux on 32-bit PowerPC
processors that implement the Signal Processing Engine (SPE), such as e500, and
uses a different ABI than standard `powerpc-unknown-linux-gnu`.
When building for other 32-bit PowerPC processors, use
`powerpc-unknown-linux-gnu` instead.

See also [Debian Wiki](https://wiki.debian.org/PowerPCSPEPort) for details on
this platform, and [ABI reference](https://web.archive.org/web/20120608163804/https://www.power.org/resources/downloads/Power-Arch-32-bit-ABI-supp-1.0-Unified.pdf)
for details on SPE ABI.

Note that support for PowerPC SPE by GCC was [removed in GCC 9](https://gcc.gnu.org/gcc-8/changes.html),
so recent GCC cannot be used as linker/compiler for this target.

## Target maintainers

There are currently no formally documented target maintainers.
