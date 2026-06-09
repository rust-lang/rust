# mipsel-sony-psx

**Tier: 3**

Sony PlayStation 1 (psx)

## Designated Developer

[@ayrtonm](https://github.com/ayrtonm)

## Requirements

This target is cross-compiled.
It has no special requirements for the host.

## Building

The target can be built by enabling it for a `rustc` build:

```toml
[build]
build-stage = 1
target = ["mipsel-sony-psx"]
```

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

Currently there is no support to run the rustc test suite for this target.

## Building Rust programs

Since it is Tier 3, rust doesn't ship pre-compiled artifacts for this target.

Just use the `build-std` nightly cargo feature to build the `core` and `alloc` libraries:
```shell
cargo build -Zbuild-std=core,alloc --target mipsel-sony-psx
```

The command above generates an ELF. To generate binaries in the PSEXE format that emulators run, you can use [cargo-psx](https://github.com/ayrtonm/psx-sdk-rs#readme):

```shell
cargo psx build
```

or use `-Clink-arg=--oformat=binary` to produce a flat binary.
