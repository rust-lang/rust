# `*-apple-tvos`
- aarch64-apple-tvos
- x86_64-apple-tvos

**Tier: 3**

Apple tvOS targets:
- Apple tvOS on aarch64
- Apple tvOS Simulator on x86_64

## Target maintainers

* [@thomcc](https://github.com/thomcc)

## Requirements

These targets are cross-compiled. You will need appropriate versions of Xcode
and the SDKs for tvOS (`AppleTVOS.sdk`) and/or the tvOS Simulator
(`AppleTVSimulator.sdk`) to build a toolchain and target these platforms.

The targets support most (see below) of the standard library including the
allocator to the best of my knowledge, however they are very new, not yet
well-tested, and it is possible that there are various bugs.

In theory we support back to tvOS version 7.0, although the actual minimum
version you can target may be newer than this, for example due to the versions
of Xcode and your SDKs.

As with the other Apple targets, `rustc` respects the common environment
variables used by Xcode to configure this, in this case
`TVOS_DEPLOYMENT_TARGET`.

#### Incompletely supported library functionality

As mentioned, "most" of the standard library is supported, which means that some portions
are known to be unsupported. The following APIs are currently known to have
missing or incomplete support:

- `std::process::Command`'s API will return an error if it is configured in a
  manner which cannot be performed using `posix_spawn` -- this is because the
  more flexible `fork`/`exec`-based approach is prohibited on these platforms in
  favor of `posix_spawn{,p}` (which still probably will get you rejected from
  app stores, so is likely sideloading-only). A concrete set of cases where this
  will occur is difficult to enumerate (and would quickly become stale), but in
  some cases it may be worked around by tweaking the manner in which `Command`
  is invoked.

## Building the target

The targets can be built by enabling them for a `rustc` build in `config.toml`, by adding, for example:

```toml
[build]
build-stage = 1
target = ["aarch64-apple-tvos", "x86_64-apple-tvos"]
```

It's possible that cargo under `-Zbuild-std` may also be used to target them.

## Building Rust programs

*Note: Building for this target requires the corresponding TVOS SDK, as provided by Xcode.*

Rust programs can be built for these targets

```text
$ rustc --target aarch64-apple-tvos your-code.rs
...
$ rustc --target x86_64-apple-tvos your-code.rs
```

## Testing

There is no support for running the Rust or standard library testsuite on tvOS
or the simulators at the moment. Testing has mostly been done manually with
builds of static libraries called from Xcode or a simulator.

It hopefully will be possible to improve this in the future.

## Cross-compilation toolchains and C code

This target can be cross-compiled from x86_64 or aarch64 macOS hosts.

Other hosts are not supported for cross-compilation, but might work when also
providing the required Xcode SDK.
