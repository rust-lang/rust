# `large-data-threshold`

-----------------------

This flag controls the threshold for static data to be placed in large data
sections when using the `medium` code model on x86-64.

When using `-Ccode-model=medium`, static data smaller than this threshold will
use RIP-relative addressing (32-bit offsets), while larger data will use
absolute 64-bit addressing. This allows the compiler to generate more efficient
code for smaller data while still supporting data larger than 2GB.

The default threshold is 65536 bytes (64KB) if not specified.

## Example

```sh
rustc -Ccode-model=medium -Zlarge-data-threshold=1024 main.rs
```

This sets the threshold to 1KB, meaning only data smaller than 1024 bytes will
use RIP-relative addressing.

## Platform Support

This flag is only effective on x86-64 targets when using `-Ccode-model=medium`.
On other architectures or with other code models, this flag has no effect.
