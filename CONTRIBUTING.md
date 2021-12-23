# Contribution Guide

If you want to hack on miri yourself, great!  Here are some resources you might
find useful.

## Getting started

Check out the issues on this GitHub repository for some ideas. In particular,
look for the green `E-*` labels which mark issues that should be rather
well-suited for onboarding. For more ideas or help with hacking on Miri, you can
contact us (`oli-obk` and `RalfJ`) on the [Rust Zulip].

[Rust Zulip]: https://rust-lang.zulipchat.com

## Preparing the build environment

Miri heavily relies on internal and unstable rustc interfaces to execute MIR,
which means it is important that you install a version of rustc that Miri
actually works with.

The `rust-version` file contains the commit hash of rustc that Miri is currently
tested against. Other versions will likely not work. After installing
[`rustup-toolchain-install-master`], you can run the following command to
install that exact version of rustc as a toolchain:
```
./rustup-toolchain
```
This will set up a rustup toolchain called `miri` and set it as an override for
the current directory.

[`rustup-toolchain-install-master`]: https://github.com/kennytm/rustup-toolchain-install-master

## Building and testing Miri

Invoking Miri requires getting a bunch of flags right and setting up a custom
sysroot with xargo. The `miri` script takes care of that for you. With the
build environment prepared, compiling Miri is just one command away:

```
./miri build
```

Run `./miri` without arguments to see the other commands our build tool
supports.

### Testing the Miri driver

The Miri driver compiled from `src/bin/miri.rs` is the "heart" of Miri: it is
basically a version of `rustc` that, instead of compiling your code, runs it.
It accepts all the same flags as `rustc` (though the ones only affecting code
generation and linking obviously will have no effect) [and more][miri-flags].

[miri-flags]: README.md#miri--z-flags-and-environment-variables

For example, you can (cross-)run the driver on a particular file by doing

```sh
./miri run tests/run-pass/format.rs
./miri run tests/run-pass/hello.rs --target i686-unknown-linux-gnu
```

and you can (cross-)run the entire test suite using:

```
./miri test
MIRI_TEST_TARGET=i686-unknown-linux-gnu ./miri test
```

`./miri test FILTER` only runs those tests that contain `FILTER` in their
filename (including the base directory, e.g. `./miri test fail` will run all
compile-fail tests).

You can get a trace of which MIR statements are being executed by setting the
`MIRI_LOG` environment variable.  For example:

```sh
MIRI_LOG=info ./miri run tests/run-pass/vec.rs
```

Setting `MIRI_LOG` like this will configure logging for Miri itself as well as
the `rustc_middle::mir::interpret` and `rustc_mir::interpret` modules in rustc. You
can also do more targeted configuration, e.g. the following helps debug the
stacked borrows implementation:

```sh
MIRI_LOG=rustc_mir::interpret=info,miri::stacked_borrows ./miri run tests/run-pass/vec.rs
```

In addition, you can set `MIRI_BACKTRACE=1` to get a backtrace of where an
evaluation error was originally raised.

### Testing `cargo miri`

Working with the driver directly gives you full control, but you also lose all
the convenience provided by cargo. Once your test case depends on a crate, it
is probably easier to test it with the cargo wrapper. You can install your
development version of Miri using

```
./miri install
```

and then you can use it as if it was installed by `rustup`.  Make sure you use
the same toolchain when calling `cargo miri` that you used when installing Miri!

There's a test for the cargo wrapper in the `test-cargo-miri` directory; run
`./run-test.py` in there to execute it. Like `./miri test`, this respects the
`MIRI_TEST_TARGET` environment variable to execute the test for another target.

### Using a modified standard library

Miri re-builds the standard library into a custom sysroot, so it is fairly easy
to test Miri against a modified standard library -- you do not even have to
build Miri yourself, the Miri shipped by `rustup` will work. All you have to do
is set the `MIRI_LIB_SRC` environment variable to the `library` folder of a
`rust-lang/rust` repository checkout. Note that changing files in that directory
does not automatically trigger a re-build of the standard library; you have to
clear the Miri build cache manually (on Linux, `rm -rf ~/.cache/miri`).

## Configuring `rust-analyzer`

To configure `rust-analyzer` and VS Code for working on Miri, save the following
to `.vscode/settings.json` in your local Miri clone:

```json
{
    "rust-analyzer.checkOnSave.overrideCommand": [
        "./miri",
        "check",
        "--message-format=json"
    ],
    "rust-analyzer.rustfmt.extraArgs": [
        "+nightly"
    ],
    "rust-analyzer.rustcSource": "discover",
    "rust-analyzer.linkedProjects": [
        "./Cargo.toml",
        "./cargo-miri/Cargo.toml"
    ]
}
```

> #### Note
>
> If you are [building Miri with a locally built rustc][], set
> `rust-analyzer.rustcSource` to the relative path from your Miri clone to the
> root `Cargo.toml` of the locally built rustc. For example, the path might look
> like `../rust/Cargo.toml`.

See the rustc-dev-guide's docs on ["Configuring `rust-analyzer` for `rustc`"][rdg-r-a]
for more information about configuring VS Code and `rust-analyzer`.

[rdg-r-a]: https://rustc-dev-guide.rust-lang.org/building/suggested.html#configuring-rust-analyzer-for-rustc

## Advanced topic: other build environments

We described above the simplest way to get a working build environment for Miri,
which is to use the version of rustc indicated by `rustc-version`. But
sometimes, that is not enough.

### Updating `rustc-version`

The `rustc-version` file is regularly updated to keep Miri close to the latest
version of rustc. Usually, new contributors do not have to worry about this. But
sometimes a newer rustc is needed for a patch, and sometimes Miri needs fixing
for changes in rustc. In both cases, `rustc-version` needs updating.

To update the `rustc-version` file and install the latest rustc, you can run:
```
./rustup-toolchain HEAD
```

Now edit Miri until `./miri test` passes, and submit a PR. Generally, it is
preferred to separate updating `rustc-version` and doing what it takes to get
Miri working again, from implementing new features that rely on the updated
rustc. This avoids blocking all Miri development on landing a big PR.

### Building Miri with a locally built rustc

[building Miri with a locally built rustc]: #building-miri-with-a-locally-built-rustc

A big part of the Miri driver lives in rustc, so working on Miri will sometimes
require using a locally built rustc. The bug you want to fix may actually be on
the rustc side, or you just need to get more detailed trace of the execution
than what is possible with release builds -- in both cases, you should develop
miri against a rustc you compiled yourself, with debug assertions (and hence
tracing) enabled.

The setup for a local rustc works as follows:
```sh
# Clone the rust-lang/rust repo.
git clone https://github.com/rust-lang/rust rustc
cd rustc
# Create a config.toml with defaults for working on miri.
./x.py setup compiler
 # Now edit `config.toml` and under `[rust]` set `debug-assertions = true`.

# Build a stage 2 rustc, and build the rustc libraries with that rustc.
# This step can take 30 minutes or more.
./x.py build --stage 2 compiler/rustc
# If you change something, you can get a faster rebuild by doing
./x.py build --keep-stage 0 --stage 2 compiler/rustc
# You may have to change the architecture in the next command
rustup toolchain link stage2 build/x86_64-unknown-linux-gnu/stage2
# Now cd to your Miri directory, then configure rustup
rustup override set stage2
```

For more information about building and configuring a local compiler,
see <https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html>.

With this, you should now have a working development setup! See
[above](#building-and-testing-miri) for how to proceed working on Miri.
