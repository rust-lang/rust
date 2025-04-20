# `armv5te-none-eabi`

**Tier: 3**

Bare-metal target for any cpu in the Armv5TE architecture family, supporting
ARM/Thumb code interworking (aka `A32`/`T32`), with `A32` code as the default code
generation.

The `thumbv5te-none-eabi` target is the same as this one, but the instruction set defaults to `T32`.

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

## Target Maintainers

[@QuinnPainter](https://github.com/QuinnPainter)

## Testing

This is a cross-compiled target that you will need to emulate during testing.

Because this is a device-agnostic target, and the exact emulator that you'll
need depends on the specific device you want to run your code on.

For example, when programming for the DS, you can use one of the several
available DS emulators, such as [melonDS](https://melonds.kuribo64.net/).
