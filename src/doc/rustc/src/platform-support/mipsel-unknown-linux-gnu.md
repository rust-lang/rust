# `mipsel-unknown-linux-gnu`

**Tier: 3**

Little-endian 32 bit MIPS for Linux with `glibc.

## Target maintainers

[@LukasWoodtli](https://github.com/LukasWoodtli)

## Requirements

The target supports std on Linux. Host tools are supported but not tested.


## Building the target

For cross compilation the GNU C compiler for the mipsel architecture needs to
be installed. On Ubuntu install the packets: `gcc-mipsel-linux-gnu` and
`g++-mipsel-linux-gnu`.

Add `mipsel-unknown-linux-gnu` as `target` list in `config.toml`.

## Building Rust programs

Rust does not ship pre-compiled artifacts for this target. To compile for
this target, you will need to build Rust with the target enabled (see
"Building the target" above).
