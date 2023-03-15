# Contribution Guide

If you want to hack on Miri yourself, great!  Here are some resources you might
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
./miri toolchain
```
This will set up a rustup toolchain called `miri` and set it as an override for
the current directory.

You can also create a `.auto-everything` file (contents don't matter, can be empty), which
will cause any `./miri` command to automatically call `./miri toolchain`, `clippy` and `rustfmt`
for you. If you don't want all of these to happen, you can add individual `.auto-toolchain`,
`.auto-clippy` and `.auto-fmt` files respectively.

[`rustup-toolchain-install-master`]: https://github.com/kennytm/rustup-toolchain-install-master

## Building and testing Miri

Invoking Miri requires getting a bunch of flags right and setting up a custom
sysroot. The `miri` script takes care of that for you. With the
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
./miri run tests/pass/format.rs
./miri run tests/pass/hello.rs --target i686-unknown-linux-gnu
```

and you can (cross-)run the entire test suite using:

```
./miri test
MIRI_TEST_TARGET=i686-unknown-linux-gnu ./miri test
```

If your target doesn't support libstd, you can run miri with

```
MIRI_NO_STD=1 MIRI_TEST_TARGET=thumbv7em-none-eabihf ./miri test tests/fail/alloc/no_global_allocator.rs
MIRI_NO_STD=1 ./miri run tests/pass/no_std.rs --target thumbv7em-none-eabihf
```

to avoid attempting (and failing) to build libstd. Note that almost no tests will pass
this way, but you can run individual tests.

`./miri test FILTER` only runs those tests that contain `FILTER` in their
filename (including the base directory, e.g. `./miri test fail` will run all
compile-fail tests).

You can get a trace of which MIR statements are being executed by setting the
`MIRI_LOG` environment variable.  For example:

```sh
MIRI_LOG=info ./miri run tests/pass/vec.rs
```

Setting `MIRI_LOG` like this will configure logging for Miri itself as well as
the `rustc_middle::mir::interpret` and `rustc_mir::interpret` modules in rustc. You
can also do more targeted configuration, e.g. the following helps debug the
stacked borrows implementation:

```sh
MIRI_LOG=rustc_mir::interpret=info,miri::stacked_borrows ./miri run tests/pass/vec.rs
```

In addition, you can set `MIRI_BACKTRACE=1` to get a backtrace of where an
evaluation error was originally raised.

### UI testing

We use ui-testing in Miri, meaning we generate `.stderr` and `.stdout` files for the output
produced by Miri. You can use `./miri bless` to automatically (re)generate these files when
you add new tests or change how Miri presents certain output.

Note that when you also use `MIRIFLAGS` to change optimizations and similar, the ui output
will change in unexpected ways. In order to still be able
to run the other checks while ignoring the ui output, use `MIRI_SKIP_UI_CHECKS=1 ./miri test`.

For more info on how to configure ui tests see [the documentation on the ui test crate][ui_test]

[ui_test]: ui_test/README.md

### Testing `cargo miri`

Working with the driver directly gives you full control, but you also lose all
the convenience provided by cargo. Once your test case depends on a crate, it
is probably easier to test it with the cargo wrapper. You can install your
development version of Miri using

```
./miri install
```

and then you can use it as if it was installed by `rustup` as a component of the
`miri` toolchain. Note that the `miri` and `cargo-miri` executables are placed
in the `miri` toolchain's sysroot to prevent conflicts with other toolchains.
The Miri binaries in the `cargo` bin directory (usually `~/.cargo/bin`) are managed by rustup.

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
clear the Miri build cache manually (on Linux, `rm -rf ~/.cache/miri`;
on Windows, `rmdir /S "%LOCALAPPDATA%\rust-lang\miri\cache"`;
and on macOS, `rm -rf ~/Library/Caches/org.rust-lang.miri`).

### Benchmarking

Miri comes with a few benchmarks; you can run `./miri bench` to run them with the locally built
Miri. Note: this will run `./miri install` as a side-effect. Also requires `hyperfine` to be
installed (`cargo install hyperfine`).

## Configuring `rust-analyzer`

To configure `rust-analyzer` and VS Code for working on Miri, save the following
to `.vscode/settings.json` in your local Miri clone:

```json
{
    "rust-analyzer.rustc.source": "discover",
    "rust-analyzer.linkedProjects": [
        "./Cargo.toml",
        "./cargo-miri/Cargo.toml"
    ],
    "rust-analyzer.checkOnSave.overrideCommand": [
        "env",
        "MIRI_AUTO_OPS=no",
        "./miri",
        "cargo",
        "clippy", // make this `check` when working with a locally built rustc
        "--message-format=json"
    ],
    // Contrary to what the name suggests, this also affects proc macros.
    "rust-analyzer.cargo.buildScripts.overrideCommand": [
        "env",
        "MIRI_AUTO_OPS=no",
        "./miri",
        "cargo",
        "check",
        "--message-format=json",
    ],
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

## Advanced topic: Working on Miri in the rustc tree

We described above the simplest way to get a working build environment for Miri,
which is to use the version of rustc indicated by `rustc-version`. But
sometimes, that is not enough.

A big part of the Miri driver is shared with rustc, so working on Miri will
sometimes require also working on rustc itself. In this case, you should *not*
work in a clone of the Miri repository, but in a clone of the
[main Rust repository](https://github.com/rust-lang/rust/). There is a copy of
Miri located at `src/tools/miri` that you can work on directly. A maintainer
will eventually sync those changes back into this repository.

When working on Miri in the rustc tree, here's how you can run tests:

```
./x.py test miri --stage 0
```

`--bless` will work, too.

You can also directly run Miri on a Rust source file:

```
./x.py run miri --stage 0 --args src/tools/miri/tests/pass/hello.rs
```

## Advanced topic: Syncing with the rustc repo

We use the [`josh` proxy](https://github.com/josh-project/josh) to transmit changes between the
rustc and Miri repositories.

```sh
cargo +stable install josh-proxy --git https://github.com/josh-project/josh --tag r22.12.06
josh-proxy --local=$HOME/.cache/josh --remote=https://github.com --no-background
```

This uses a directory `$HOME/.cache/josh` as a cache, to speed up repeated pulling/pushing.

To make josh push via ssh instead of https, you can add the following to your `.gitconfig`:

```toml
[url "git@github.com:"]
    pushInsteadOf = https://github.com/
```

### Importing changes from the rustc repo

Josh needs to be running, as described above.
We assume we start on an up-to-date master branch in the Miri repo.

```sh
# Fetch and merge rustc side of the history. Takes ca 5 min the first time.
# This will also update the 'rustc-version' file.
./miri rustc-pull
# Update local toolchain and apply formatting.
./miri toolchain && ./miri fmt
git commit -am "rustup"
```

Now push this to a new branch in your Miri fork, and create a PR. It is worth
running `./miri test` locally in parallel, since the test suite in the Miri repo
is stricter than the one on the rustc side, so some small tweaks might be
needed.

### Exporting changes to the rustc repo

Keep in mind that pushing is the most complicated job that josh has to do --
pulling just filters the rustc history, but pushing needs to construct a new
rustc history that would filter to the given Miri history! To avoid problems, it
is a good idea to always pull immediately before you push. In particular, you
should never do two josh pushes without an intermediate pull; that can lead to
duplicated commits.

Josh needs to be running, as described above. We will use the josh proxy to push
to your fork of rustc. Run the following in the Miri repo, assuming we are on an
up-to-date master branch:

```sh
# Push the Miri changes to your rustc fork (substitute your github handle for YOUR_NAME).
./miri rustc-push YOUR_NAME miri
```

This will create a new branch called 'miri' in your fork, and the output should
include a link to create a rustc PR that will integrate those changes into the
main repository.
