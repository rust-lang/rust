# `armv6-none-eabi*` and `thumbv6-none-eabi`

* **Tier: 3**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for any cpu in the Armv6 architecture family, supporting
ARM/Thumb code interworking (aka `Arm`/`Thumb`), with `Arm` code as the default
code generation. The most common processor family using the Armv6 architecture
is the ARM11, which includes the ARM1176JZF-S used in the original Raspberry Pi
and in the Raspberry Pi Zero.

This target assumes your processor has the Armv6K extensions, as basically all
Armv6 processors do[^1]. The Armv6K extension adds the `LDREXB` and `STREXB`
instructions required to efficiently implement CAS on the [`AtomicU8`] and
[`AtomicI8`] types.

The `thumbv6-none-eabi` target is the same as this one, but the instruction set
defaults to `Thumb`. Note that this target only supports the old Thumb-1
instruction set, not the later Thumb-2 instruction set that was added in the
Armv6T2 extension. Note that the Thumb-1 instruction set does not support
atomics.

The `armv6-none-eabihf` target uses the EABIHF hard-float ABI, and requires an
FPU - it assumes a VFP2D16 FPU is present. The FPU is not available from Thumb
mode so there is no `thumbv6-none-eabihf` target.

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

[`AtomicU8`]: https://docs.rust-lang.org/stable/core/sync/atomic/struct.AtomicU8.html
[`AtomicI8`]: https://docs.rust-lang.org/stable/core/sync/atomic/struct.AtomicI8.html

## Target Maintainers

[@thejpster](https://github.com/thejpster)

[^1]: The only ARMv6 processor without the Armv6k extensions is the first (r0)
revision of the ARM1136 - in the unlikely event you have a chip with one of
these processors, use the ARMv5TE target instead.
