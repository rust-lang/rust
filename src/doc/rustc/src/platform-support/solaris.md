# sparcv9-sun-solaris
# x86_64-pc-solaris

**Tier: 2**

Rust for Solaris operating system.

## Target maintainers

[@psumbera](https://github.com/psumbera)

## Requirements

Binary built for this target is expected to run on sparcv9 or x86_64, and Solaris 11.4.

## Testing

For testing you can download Oracle Solaris 11.4 CBE release from:

  https://www.oracle.com/uk/solaris/solaris11/downloads/solaris-downloads.html

Solaris CBE release is also available for GitHub CI:

  https://github.com/vmactions/solaris-vm

Latest Solaris 11.4 SRU can be tested at Compile farm project:

  https://portal.cfarm.net/machines/list/ (cfarm215, cfarm215)

There are no official Rust binaries for Solaris available for Rustup yet. But you can eventually download unofficial from:

  https://github.com/psumbera/solaris-rust
