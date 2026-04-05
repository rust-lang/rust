# Changelog

## v0.6.18

 - Fixed build for ARMv4t and ARMv5te thumb mode.

## v0.6.17

 - Updated syscall lists to Linux 6.8.

## v0.6.16

 - ~~Fixed build for ARMv4t and ARMv5te thumb mode.~~ (Actually fixed in v0.6.18)

## v0.6.15

 - Added `SysnoMap::into_iter`.
 - Added `SysnoSet::into_iter`.
 - Removed unnecessary build dependency on `cc`.
 - aarch64: Fixed a doc test

## v0.6.14

 - Updated to syscalls from Linux v6.5.

## v0.6.13

 - armv7: Added support for armv7-linux-androideabi targets.

## v0.6.12

 - riscv: Added the `riscv_flush_icache` syscall
 - powerpc64: Fixed a typo preventing the `asm_experimental_arch` feature from
   being enabled.

## v0.6.11

 - Added riscv32 and riscv64 support.

## v0.6.10

 - Added `SysnoMap::init_all`.
 - Added support for ARM thumb mode. Compilation would fail in this case.
   Requires the use of `build.rs` (or using a nightly compiler).

## v0.6.9

 - Added `SysnoMap` for mapping syscall numbers to a type `T`.

## v0.6.8

 - aarch64: Removed bogus `arch_specific_syscall`.
 - x86: Fixed `int 0x80` syntax for non-LLVM rustc backends.
 - Added `SysnoSet::is_empty`.
 - Minor documentation fixes.

## v0.6.7

 - Fixed missing aarch64 syscalls

## v0.6.6

 - Added aarch64 support.

## v0.6.5

 - Renamed `with-serde` feature to just `serde`. The `with-serde` feature will
   be removed in the next major release.
 - Implemented `Serialize` and `Deserialize` for `SysnoSet`.

## v0.6.4

 - Implemented `Default`, `BitOr`, and `BitOrAssign` for `SysnoSet`.

## v0.6.3

 - Added features to expose the syscall tables of other architectures besides
   the target architecture. There is one feature per architecture and have the
   same name. For example, if the target architecture is `x86-64` and we also
   want the syscall table for `x86`, the `x86` feature can be enabled. Then,
   `syscalls::x86::Sysno` will be exposed.
 - Added the `all` feature, which enables the syscall tables for all
   architectures.
 - Added the `full` feature, which enables all current and future features for
   the crate.
 - Added man page links for all syscalls. Since these are generated, some links
   may be broken.

## v0.6.2

 - Added `SysnoSet` for constructing sets of syscalls. It uses a bitset under
   the hood and provides constant-time lookup and insertion of `Sysno`s.
 - Fixed `Sysno::len()` returning the wrong value for architectures with large
   syscall offsets.
 - Deprecated `Sysno::len()`. Use `Sysno::table_size()` instead. This will be
   removed in the next major version.

## v0.6.1

 - Exposed `syscalls::raw::*` to allow avoidance of the `Result` return type.
   This makes it cleaner to call syscalls like `gettid` that are guaranteed to
   never fail.

## v0.6.0

 - Removed `build.rs` and switched to Rust's inline assembly syntax. This should
   enable better codegen, including the ability to have syscalls get inlined.
 - **Breaking**: Architectures besides `arm`, `x86`, and `x86-64` now require
   nightly.
 - **Breaking**: Removed top-level `SYS_` constants. Just use the `Sysno` enum
   instead.

## v0.5.0

This is a major breaking change from v0.4.

 - Changed all syscalls to take and return `usize` instead of `i64` or `u64`.
   This fixes calling syscalls on 32-bit architectures.
 - Fixed syscall offsets for mips and mips64.
 - Added CI tests for more than just `x86_64`.

## v0.4.2

 - Made `ErrnoSentinel` public.

## v0.4.1

 - Added the ability to invoke syscalls for all architectures except `aarch64`,
   `sparc`, and `sparc64`.
 - Fixed std-dependent Errno trait impls not getting compiled.
 - Made `syscalls::arch::{x86, x86_64, ...}` private.

## v0.4.0

This is a major breaking change from v0.3. You can fix most compilation errors
by simply doing `s/SyscallNo::SYS_/Sysno::/g`.

 - Created this changelog.
 - Renamed `SyscallNo::SYS_*` to `Sysno::*`.
 - Added `Errno` for more Rustic error handling.
 - Changed the `syscalls-gen` script to grab Linux headers from GitHub.
 - Added more architecture support for the syscall table. Issuing syscalls is
   still limited to x86-64, however.
