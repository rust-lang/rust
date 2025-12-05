# Contribution Guide

If you want to hack on Miri yourself, great!  Here are some resources you might
find useful.

## Getting started

Check out the issues on this GitHub repository for some ideas. In particular,
look for the green `E-*` labels which mark issues that should be rather
well-suited for onboarding. For more ideas or help with hacking on Miri, you can
contact us on the [Rust Zulip]. See the [Rust website](https://www.rust-lang.org/governance/teams/compiler#team-miri)
for a list of Miri maintainers.

[Rust Zulip]: https://rust-lang.zulipchat.com

### PR review process

When you get a review, please take care of the requested changes in new commits. Do not amend
existing commits. Generally avoid force-pushing. The only time you should force push is when there
is a conflict with the master branch (in that case you should rebase across master, not merge), and
all the way at the end of the review process when the reviewer tells you that the PR is done and you
should squash the commits. (All this is to work around the fact that Github is quite bad at
dealing with force pushes and does not support `git range-diff`.)

The recommended way to squash commits is to use `./miri squash`, which will make everything into a
single commit. You will be asked for the commit message; please ensure it describes the entire PR.
You can also use `git rebase` manually if you need more control (e.g. if there should be more than
one commit at the end), but then please use `--keep-base` to ensure the PR remains based on the same
upstream commit.

Most PRs bounce back and forth between the reviewer and the author several times, so it is good to
keep track of who is expected to take the next step. We are using the `S-waiting-for-review` and
`S-waiting-for-author` labels for that. If a reviewer asked you to do some changes and you think
they are all taken care of, post a comment saying `@rustbot ready` to mark a PR as ready for the
next round of review.

### Larger-scale contributions

If you are thinking about making a larger-scale contribution -- in particular anything that needs
more than can reasonably fit in a single PR to be feature-complete -- then please talk to us before
writing significant amounts of code. Generally, we will ask that you follow a three-step "project"
process for such contributions:

1. Clearly define the **goal** of the project. This defines the scope of the project, i.e. which
   part of which APIs should be supported. If this involves functions that expose a big API surface
   with lots of flags, the project may want to support only a tiny subset of flags; that should be
   documented. A good way to express the goal is with one or more test cases that Miri should be
   able to successfully execute when the project is completed. It is a good idea to get feedback
   from team members already at this stage to ensure that the project is reasonably scoped and
   aligns with our interests.
2. Make a **design** for how to realize the goal. A larger project will likely have to do global
   changes to Miri, like adding new global state to the `Machine` type or new methods to the
   `FileDescription` trait. Often we have to iterate on those changes, which can quite substantially
   change how the final implementation looks like.

    The design should be reasonably concrete, i.e. for new global state or methods the corresponding
   Rust types and method signatures should be spelled out. We realize that it can be hard to make a
   design without doing implementation work, in particular if you are not yet familiar with the
   codebase. Doing draft implementations in phase 2 of this process is perfectly fine, just please
   be aware that we might request fundamental changes that can require significantly reworking what
   you already did. If you open a PR in this stage, please clearly indicate that this project is
   still in the design stage.

3. Finish the **implementation** and have it reviewed.

This process is largely informal, and its primary goal is to more clearly communicate expectations.
Please get in touch with us if you have any questions!

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

Tests in ``pass-dep`` need to be run using ``./miri run --dep <filename>``.  
For example:
```sh
./miri run --dep tests/pass-dep/shims/libc-fs.rs
```

You can (cross-)run the entire test suite using:

```sh
./miri test
./miri test --target i686-unknown-linux-gnu
```

`./miri test FILTER` only runs those tests that contain `FILTER` in their filename (including the
base directory, e.g. `./miri test fail` will run all compile-fail tests). Multiple filters are
supported: `./miri test FILTER1 FILTER2` runs all tests that contain either string.

#### Fine grained logging

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

Note that you will only get `info`, `warn` or `error` messages if you use a prebuilt compiler.
In order to get `debug` and `trace` level messages, you need to build miri with a locally built
compiler that has `debug=true` set in `bootstrap.toml`.

#### Debugging error messages

You can set `MIRI_BACKTRACE=1` to get a backtrace of where an
evaluation error was originally raised.

#### Tracing

You can generate a Chrome trace file from a Miri execution by passing `--features=tracing` during the
build and then setting `MIRI_TRACING=1` when running Miri. This will generate a `.json` file that
you can visualize in [Perfetto](https://ui.perfetto.dev/). For example:

```sh
MIRI_TRACING=1 ./miri run --features=tracing tests/pass/hello.rs
```

### UI testing

We use ui-testing in Miri, meaning we generate `.stderr` and `.stdout` files for the output
produced by Miri. You can use `./miri test --bless` to automatically (re)generate these files when
you add new tests or change how Miri presents certain output.

Note that when you also use `MIRIFLAGS` to change optimizations and similar, the ui output
will change in unexpected ways. In order to still be able
to run the other checks while ignoring the ui output, use `MIRI_SKIP_UI_CHECKS=1 ./miri test`.

For more info on how to configure ui tests see [the documentation on the ui test crate][ui_test]

[ui_test]: https://github.com/oli-obk/ui_test/blob/main/README.md

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

There's a test for the cargo wrapper in the `test-cargo-miri` directory; run `./run-test.py` in
there to execute it. You can pass `--target` to execute the test for another target.

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

To compare the benchmark results with a baseline, do the following:
- Before applying your changes, run `./miri bench --save-baseline=baseline.json`.
- Then do your changes.
- Then run `./miri bench --load-baseline=baseline.json`; the results will include
  a comparison with the baseline.

You can run only some of the benchmarks by listing them, e.g. `./miri bench mse`.
The names refer to the folders in `bench-cargo-miri`.

## Configuring `rust-analyzer`

To configure `rust-analyzer` and the IDE for working on Miri, copy one of the provided
configuration files according to the instructions below. You can also set up a symbolic
link to keep the configuration in sync with our recommendations.

### Visual Studio Code

Copy [`etc/rust_analyzer_vscode.json`] to `.vscode/settings.json` in the project root directory.

[`etc/rust_analyzer_vscode.json`]: https://github.com/rust-lang/miri/blob/master/etc/rust_analyzer_vscode.json

### Helix

Copy [`etc/rust_analyzer_helix.toml`] to `.helix/languages.toml` in the project root directory.

Since working on Miri requires a custom toolchain, and Helix requires the language server
to be installed with the toolchain, you have to run `./miri toolchain -c rust-analyzer`
when installing the Miri toolchain. Alternatively, set the `RUSTUP_TOOLCHAIN` environment variable according to
[the documentation](https://rust-analyzer.github.io/manual.html#toolchain).

[`etc/rust_analyzer_helix.toml`]: https://github.com/rust-lang/miri/blob/master/etc/rust_analyzer_helix.toml

### Zed

Copy [`etc/rust_analyzer_zed.json`] to `.zed/settings.json` in the project root directory.

[`etc/rust_analyzer_zed.json`]: https://github.com/rust-lang/miri/blob/master/etc/rust_analyzer_zed.json

### Advanced configuration

If you are building Miri with a locally built rustc, set
`rust-analyzer.rustcSource` to the relative path from your Miri clone to the
root `Cargo.toml` of the locally built rustc. For example, the path might look
like `../rust/Cargo.toml`. In addition to that, replace `clippy` by `check`
in the `rust-analyzer.check.overrideCommand` setting.

See the rustc-dev-guide's docs on ["Configuring `rust-analyzer` for `rustc`"][rdg-r-a]
for more information about configuring the IDE and `rust-analyzer`.

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
./x.py test miri
```

`--bless` will work, too.

You can also directly run Miri on a Rust source file:

```
./x.py run miri --stage 1 --args src/tools/miri/tests/pass/hello.rs
```

## Advanced topic: Syncing with the rustc repo

We use the [`josh-sync`](https://github.com/rust-lang/josh-sync) tool to transmit changes between the
rustc and Miri repositories. You can install it as follows:

```sh
cargo install --locked --git https://github.com/rust-lang/josh-sync
```

The commands below will automatically install and manage the [Josh](https://github.com/josh-project/josh) proxy that performs the actual work.

### Importing changes from the rustc repo

*Note: this usually happens automatically, so these steps rarely have to be done by hand.*

We assume we start on an up-to-date master branch in the Miri repo.

1) First, create a branch for the pull, e.g. `git checkout -b rustup`
2) Then run the following:
```sh
# Fetch and merge rustc side of the history. Takes ca 5 min the first time.
# This will also update the `rustc-version` file.
rustc-josh-sync pull
# Update local toolchain and apply formatting.
./miri toolchain && ./miri fmt
git commit -am "rustup"
```

Now push this to a new branch in your Miri fork, and create a PR. It is worth
running `./miri test` locally in parallel, since the test suite in the Miri repo
is stricter than the one on the rustc side, so some small tweaks might be
needed.

### Exporting changes to the rustc repo

We will use the `josh-sync` tool to push to your fork of rustc. Run the following in the Miri repo,
assuming we are on an up-to-date master branch:

```sh
# Push the Miri changes to your rustc fork (substitute your github handle for YOUR_NAME).
rustc-josh-sync push miri YOUR_NAME
```

This will create a new branch called `miri` in your fork, and the output should include a link that
creates a rustc PR to integrate those changes into the main repository. If that PR has conflicts,
you need to pull rustc changes into Miri first, and then re-do the rustc push.

If this fails due to authentication problems, it can help to make josh push via ssh instead of
https. Add the following to your `.gitconfig`:

```toml
[url "git@github.com:"]
    pushInsteadOf = https://github.com/
```

## Further environment variables

The following environment variables are relevant to `./miri`:

* `CARGO` sets the binary used to execute Cargo; if none is specified, defaults to `cargo`.
* `MIRI_AUTO_OPS` indicates whether the automatic execution of rustfmt, clippy and toolchain setup
  (as controlled by the `./auto-*` files) should be skipped. If it is set to `no`, they are skipped.
  This is used to allow automated IDE actions to avoid the auto ops.
* `MIRI_LOG`, `MIRI_BACKTRACE` control logging and backtrace printing during Miri executions.
* `MIRI_TEST_THREADS` (recognized by `./miri test`) sets the number of threads to use for running
  tests. By default, the number of cores is used.
* `MIRI_SKIP_UI_CHECKS` (recognized by `./miri test`) disables checking that the `stderr` or
  `stdout` files match the actual output.

Furthermore, the usual environment variables recognized by `cargo miri` also work for `./miri`, e.g.
`MIRI_LIB_SRC`. Note that `MIRIFLAGS` is ignored by `./miri test` as each test controls the flags it
is run with.

The following environment variables are *internal* and must not be used by
anyone but Miri itself. They are used to communicate between different Miri
binaries, and as such worth documenting:

* `CARGO_EXTRA_FLAGS` is understood by `./miri` and passed to all host cargo invocations.
  It is reserved for CI usage; setting the wrong flags this way can easily confuse the script.
* `MIRI_BE_RUSTC` can be set to `host` or `target`. It tells the Miri driver to
  actually not interpret the code but compile it like rustc would. With `target`, Miri sets
  some compiler flags to prepare the code for interpretation; with `host`, this is not done.
  This environment variable is useful to be sure that the compiled `rlib`s are compatible
  with Miri.
* `MIRI_CALLED_FROM_SETUP` is set during the Miri sysroot build,
  which will re-invoke `cargo-miri` as the `rustc` to use for this build.
* `MIRI_CALLED_FROM_RUSTDOC` when set to any value tells `cargo-miri` that it is
  running as a child process of `rustdoc`, which invokes it twice for each doc-test
  and requires special treatment, most notably a check-only build before interpretation.
  This is set by `cargo-miri` itself when running as a `rustdoc`-wrapper.
* `MIRI_CWD` when set to any value tells the Miri driver to change to the given
  directory after loading all the source files, but before commencing
  interpretation. This is useful if the interpreted program wants a different
  working directory at run-time than at build-time.
* `MIRI_LOCAL_CRATES` is set by `cargo-miri` to tell the Miri driver which
  crates should be given special treatment in diagnostics, in addition to the
  crate currently being compiled.
* `MIRI_ORIG_RUSTDOC` is set and read by different phases of `cargo-miri` to remember the
  value of `RUSTDOC` from before it was overwritten.
* `MIRI_REPLACE_LIBRS_IF_NOT_TEST` when set to any value enables a hack that helps bootstrap
  run the standard library tests in Miri.
* `MIRI_TEST_TARGET` is set by `./miri test` (and `./x.py test miri`) to tell the test harness about
  the chosen target.
* `MIRI_VERBOSE` when set to any value tells the various `cargo-miri` phases to
  perform verbose logging.
* `MIRI_HOST_SYSROOT` is set by bootstrap to tell `cargo-miri` which sysroot to use for *host*
  operations.
* `RUSTC_BLESS` is set by `./miri test` (and `./x.py test miri`) to indicate bless-mode to the test
  harness.
