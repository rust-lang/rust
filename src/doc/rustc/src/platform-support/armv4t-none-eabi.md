# armv4t-none-eabi

Tier 3

Bare-metal target for any cpu in the ARMv4T architecture family, supporting
ARM/Thumb code interworking (aka `a32`/`t32`), with ARM code as the default code
generation.

In particular this supports the Game Boy Advance (GBA), but there's nothing
GBA-specific with this target, so any ARMv4T device should work fine.

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

## Target Maintainers

* [@Lokathor](https://github.com/lokathor)

## Testing

This is a cross-compiled target that you will need to emulate during testing.

Because this is a device-agnostic target, and the exact emulator that you'll
need depends on the specific device you want to run your code on.

For example, when programming for the Gameboy Advance, the
[mgba-test-runner](https://github.com/agbrs/agb) program could be used to make a
normal set of rust tests be run within the `mgba` emulator.
