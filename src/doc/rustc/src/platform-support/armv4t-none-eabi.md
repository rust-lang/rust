# armv4t-none-eabi / thumbv4t-none-eabi

* **Tier:  3**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

These two targets are part of the [`arm-none-eabi`](arm-none-eabi.md) target
group, and all the information there applies.

Both of these targets can be used on the Game Boy Advance (GBA), among other
things. On the GBA, one should usually use the `thumb` target to get the best
overall performance.

## Target Maintainers

[@Lokathor](https://github.com/lokathor)
[@corwinkuiper](https://github.com/corwinkuiper)

## Testing

This is a cross-compiled target that you will need to emulate during testing.

Because this is a device-agnostic target, and the exact emulator that you'll
need depends on the specific device you want to run your code on.

* When building for the GBA, [mgba-test-runner](https://github.com/agbrs/agb)
  can be used to make a normal set of rust tests be run within the `mgba`
  emulator.
