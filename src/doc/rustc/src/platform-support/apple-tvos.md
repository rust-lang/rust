# `*-apple-tvos`

Apple tvOS targets.

**Tier: 3**

- `aarch64-apple-tvos`: Apple tvOS on ARM64.
- `aarch64-apple-tvos-sim`: Apple tvOS Simulator on ARM64.
- `x86_64-apple-tvos`: Apple tvOS Simulator on x86_64.

## Target maintainers

- [@thomcc](https://github.com/thomcc)
- [@madsmtm](https://github.com/madsmtm)

## Requirements

These targets are cross-compiled, and require the corresponding tvOS SDK
(`AppleTVOS.sdk` or `AppleTVSimulator.sdk`), as provided by Xcode. To build the
ARM64 targets, Xcode 12 or higher is required.

The path to the SDK can be passed to `rustc` using the common `SDKROOT`
environment variable.

### OS version

The minimum supported version is tvOS 10.0, although the actual minimum version
you can target may be newer than this, for example due to the versions of Xcode
and your SDKs.

The version can be raised per-binary by changing the deployment target. `rustc`
respects the common environment variables used by Xcode to do so, in this
case `TVOS_DEPLOYMENT_TARGET`.

### Incompletely supported library functionality

The targets support most of the standard library including the allocator to the
best of my knowledge, however they are very new, not yet well-tested, and it is
possible that there are various bugs.

The following APIs are currently known to have missing or incomplete support:

- `std::process::Command`'s API will return an error if it is configured in a
  manner which cannot be performed using `posix_spawn` -- this is because the
  more flexible `fork`/`exec`-based approach is prohibited on these platforms in
  favor of `posix_spawn{,p}` (which still probably will get you rejected from
  app stores, so is likely sideloading-only). A concrete set of cases where this
  will occur is difficult to enumerate (and would quickly become stale), but in
  some cases it may be worked around by tweaking the manner in which `Command`
  is invoked.

## Building the target

The targets can be built by enabling them for a `rustc` build in
`bootstrap.toml`, by adding, for example:

```toml
[build]
build-stage = 1
target = ["aarch64-apple-tvos", "aarch64-apple-tvos-sim"]
```

Using the unstable `-Zbuild-std` with a nightly Cargo may also work.

## Building Rust programs

Rust programs can be built for these targets by specifying `--target`, if
`rustc` has been built with support for them. For example:

```console
$ rustc --target aarch64-apple-tvos your-code.rs
```

## Testing

There is no support for running the Rust or standard library testsuite at the
moment. Testing has mostly been done manually with builds of static libraries
embedded into applications called from Xcode or a simulator.

It hopefully will be possible to improve this in the future.
