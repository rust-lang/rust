# `aarch64-unknown-fuchsia` and `x86_64-unknown-fuchsia`

**Tier: 2**

[Fuchsia] is a modern open source operating system that's simple, secure,
updatable, and performant.

## Target maintainers

[@erickt](https://github.com/erickt)
[@Nashenas88](https://github.com/Nashenas88)

The up-to-date list can be also found via the
[fuchsia marker team](https://github.com/rust-lang/team/blob/master/teams/fuchsia.toml).

## Table of contents

1. [Requirements](#requirements)
1. [Walkthrough structure](#walkthrough-structure)
1. [Compiling a Rust binary targeting Fuchsia](#compiling-a-rust-binary-targeting-fuchsia)
    1. [Targeting Fuchsia with rustup and cargo](#targeting-fuchsia-with-rustup-and-cargo)
    1. [Targeting Fuchsia with a compiler built from source](#targeting-fuchsia-with-a-compiler-built-from-source)
1. [Creating a Fuchsia package](#creating-a-fuchsia-package)
    1. [Creating a Fuchsia component](#creating-a-fuchsia-component)
    1. [Building a Fuchsia package](#building-a-fuchsia-package)
1. [Publishing a Fuchsia package](#publishing-a-fuchsia-package)
    1. [Creating a Fuchsia package repository](#creating-a-fuchsia-package-repository)
    1. [Publishing Fuchsia package to repository](#publishing-fuchsia-package-to-repository)
1. [Running a Fuchsia component on an emulator](#running-a-fuchsia-component-on-an-emulator)
    1. [Starting the Fuchsia emulator](#starting-the-fuchsia-emulator)
    1. [Watching emulator logs](#watching-emulator-logs)
    1. [Serving a Fuchsia package](#serving-a-fuchsia-package)
    1. [Running a Fuchsia component](#running-a-fuchsia-component)
1. [`.gitignore` extensions](#gitignore-extensions)
1. [Testing](#testing)
    1. [Running unit tests](#running-unit-tests)
    1. [Running the compiler test suite](#running-the-compiler-test-suite)
1. [Debugging](#debugging)
    1. [`zxdb`](#zxdb)
    1. [Attaching `zxdb`](#attaching-zxdb)
    1. [Using `zxdb`](#using-zxdb)
    1. [Displaying source code in `zxdb`](#displaying-source-code-in-zxdb)

## Requirements

This target is cross-compiled from a host environment. You will need a recent
copy of the [Fuchsia SDK], which provides the tools, libraries, and binaries
required to build and link programs for Fuchsia.

Development may also be done from the [source tree].

Fuchsia targets support `std` and follow the `sysv64` calling convention on
x86_64. Fuchsia binaries use the ELF file format.

## Walkthrough structure

This walkthrough will cover:

1. Compiling a Rust binary targeting Fuchsia.
1. Building a Fuchsia package.
1. Publishing and running a Fuchsia package to a Fuchsia emulator.

For the purposes of this walkthrough, we will only target `x86_64-unknown-fuchsia`.

## Compiling a Rust binary targeting Fuchsia

Today, there are two main ways to build a Rust binary targeting Fuchsia
using the Fuchsia SDK:
1. Allow [rustup] to handle the installation of Fuchsia targets for you.
1. Build a toolchain locally that can target Fuchsia.

### Targeting Fuchsia with rustup and cargo

The easiest way to build a Rust binary targeting Fuchsia is by allowing [rustup]
to handle the installation of Fuchsia targets for you. This can be done by issuing
the following commands:

```sh
rustup target add x86_64-unknown-fuchsia
rustup target add aarch64-unknown-fuchsia
```

After installing our Fuchsia targets, we can now compile a Rust binary that targets
Fuchsia.

To create our Rust project, we can use [`cargo`][cargo] as follows:

**From base working directory**
```sh
cargo new hello_fuchsia
```

The rest of this walkthrough will take place from `hello_fuchsia`, so we can
change into that directory now:

```sh
cd hello_fuchsia
```

*Note: From this point onwards, all commands will be issued from the `hello_fuchsia/`
directory, and all `hello_fuchsia/` prefixes will be removed from references for sake of brevity.*

We can edit our `src/main.rs` to include a test as follows:

**`src/main.rs`**
```rust
fn main() {
    println!("Hello Fuchsia!");
}

#[test]
fn it_works() {
    assert_eq!(2 + 2, 4);
}
```

In addition to the standard workspace created, we will want to create a
`.cargo/config.toml` file to link necessary libraries
during compilation:

**`.cargo/config.toml`**
```txt
[target.x86_64-unknown-fuchsia]

rustflags = [
    "-Lnative=<SDK_PATH>/arch/x64/lib",
    "-Lnative=<SDK_PATH>/arch/x64/sysroot/lib"
]
```

*Note: Make sure to fill out `<SDK_PATH>` with the path to the downloaded [Fuchsia SDK].*

These options configure the following:

* `-Lnative=${SDK_PATH}/arch/${ARCH}/lib`: Link against Fuchsia libraries from
  the SDK
* `-Lnative=${SDK_PATH}/arch/${ARCH}/sysroot/lib`: Link against Fuchsia sysroot
  libraries from the SDK

In total, our new project will look like:

**Current directory structure**
```txt
hello_fuchsia/
┣━ src/
┃  ┗━ main.rs
┣━ Cargo.toml
┗━ .cargo/
   ┗━ config.toml
```

Finally, we can build our rust binary as:

```sh
cargo build --target x86_64-unknown-fuchsia
```

Now we have a Rust binary at `target/x86_64-unknown-fuchsia/debug/hello_fuchsia`,
targeting our desired Fuchsia target.

**Current directory structure**
```txt
hello_fuchsia/
┣━ src/
┃  ┗━ main.rs
┣━ target/
┃  ┗━ x86_64-unknown-fuchsia/
┃     ┗━ debug/
┃        ┗━ hello_fuchsia
┣━ Cargo.toml
┗━ .cargo/
   ┗━ config.toml
```

### Targeting Fuchsia with a compiler built from source

An alternative to the first workflow is to target Fuchsia by using
`rustc` built from source.

Before building Rust for Fuchsia, you'll need a clang toolchain that supports
Fuchsia as well. A recent version (14+) of clang should be sufficient to compile
Rust for Fuchsia.

x86-64 and AArch64 Fuchsia targets can be enabled using the following
configuration in `bootstrap.toml`:

```toml
[build]
target = ["<host_platform>", "aarch64-unknown-fuchsia", "x86_64-unknown-fuchsia"]

[rust]
lld = true

[llvm]
download-ci-llvm = false

[target.x86_64-unknown-fuchsia]
cc = "clang"
cxx = "clang++"

[target.aarch64-unknown-fuchsia]
cc = "clang"
cxx = "clang++"
```

Though not strictly required, you may also want to use `clang` for your host
target as well:

```toml
[target.<host_platform>]
cc = "clang"
cxx = "clang++"
```

By default, the Rust compiler installs itself to `/usr/local` on most UNIX
systems. You may want to install it to another location (e.g. a local `install`
directory) by setting a custom prefix in `bootstrap.toml`:

```toml
[install]
# Make sure to use the absolute path to your install directory
prefix = "<RUST_SRC_PATH>/install"
```

Next, the following environment variables must be configured. For example, using
a script we name `config-env.sh`:

```sh
# Configure this environment variable to be the path to the downloaded SDK
export SDK_PATH="<SDK path goes here>"

export CFLAGS_aarch64_unknown_fuchsia="--target=aarch64-unknown-fuchsia --sysroot=${SDK_PATH}/arch/arm64/sysroot -I${SDK_PATH}/pkg/fdio/include"
export CXXFLAGS_aarch64_unknown_fuchsia="--target=aarch64-unknown-fuchsia --sysroot=${SDK_PATH}/arch/arm64/sysroot -I${SDK_PATH}/pkg/fdio/include"
export LDFLAGS_aarch64_unknown_fuchsia="--target=aarch64-unknown-fuchsia --sysroot=${SDK_PATH}/arch/arm64/sysroot -L${SDK_PATH}/arch/arm64/lib"
export CARGO_TARGET_AARCH64_UNKNOWN_FUCHSIA_RUSTFLAGS="-C link-arg=--sysroot=${SDK_PATH}/arch/arm64/sysroot -Lnative=${SDK_PATH}/arch/arm64/sysroot/lib -Lnative=${SDK_PATH}/arch/arm64/lib"
export CFLAGS_x86_64_unknown_fuchsia="--target=x86_64-unknown-fuchsia --sysroot=${SDK_PATH}/arch/x64/sysroot -I${SDK_PATH}/pkg/fdio/include"
export CXXFLAGS_x86_64_unknown_fuchsia="--target=x86_64-unknown-fuchsia --sysroot=${SDK_PATH}/arch/x64/sysroot -I${SDK_PATH}/pkg/fdio/include"
export LDFLAGS_x86_64_unknown_fuchsia="--target=x86_64-unknown-fuchsia --sysroot=${SDK_PATH}/arch/x64/sysroot -L${SDK_PATH}/arch/x64/lib"
export CARGO_TARGET_X86_64_UNKNOWN_FUCHSIA_RUSTFLAGS="-C link-arg=--sysroot=${SDK_PATH}/arch/x64/sysroot -Lnative=${SDK_PATH}/arch/x64/sysroot/lib -Lnative=${SDK_PATH}/arch/x64/lib"
```

Finally, the Rust compiler can be built and installed:

```sh
(source config-env.sh && ./x.py install)
```

Once `rustc` is installed, we can create a new working directory to work from,
`hello_fuchsia` along with `hello_fuchsia/src`:

```sh
mkdir hello_fuchsia
cd hello_fuchsia
mkdir src
```

*Note: From this point onwards, all commands will be issued from the `hello_fuchsia/`
directory, and all `hello_fuchsia/` prefixes will be removed from references for sake of brevity.*

There, we can create a new file named `src/hello_fuchsia.rs`:

**`src/hello_fuchsia.rs`**
```rust
fn main() {
    println!("Hello Fuchsia!");
}

#[test]
fn it_works() {
    assert_eq!(2 + 2, 4);
}
```

**Current directory structure**
```txt
hello_fuchsia/
┗━ src/
    ┗━ hello_fuchsia.rs
```

Using your freshly installed `rustc`, you can compile a binary for Fuchsia using
the following options:

* `--target x86_64-unknown-fuchsia`/`--target aarch64-unknown-fuchsia`: Targets the Fuchsia
  platform of your choice
* `-Lnative ${SDK_PATH}/arch/${ARCH}/lib`: Link against Fuchsia libraries from
  the SDK
* `-Lnative ${SDK_PATH}/arch/${ARCH}/sysroot/lib`: Link against Fuchsia sysroot
  libraries from the SDK

Putting it all together:

```sh
# Configure these for the Fuchsia target of your choice
TARGET_ARCH="<x86_64-unknown-fuchsia|aarch64-unknown-fuchsia>"
ARCH="<x64|aarch64>"

rustc \
    --target ${TARGET_ARCH} \
    -Lnative=${SDK_PATH}/arch/${ARCH}/lib \
    -Lnative=${SDK_PATH}/arch/${ARCH}/sysroot/lib \
    --out-dir bin src/hello_fuchsia.rs
```

**Current directory structure**
```txt
hello_fuchsia/
┣━ src/
┃   ┗━ hello_fuchsia.rs
┗━ bin/
   ┗━ hello_fuchsia
```

## Creating a Fuchsia package

Before moving on, double check your directory structure:

**Current directory structure**
```txt
hello_fuchsia/
┣━ src/                         (if using rustc)
┃   ┗━ hello_fuchsia.rs         ...
┣━ bin/                         ...
┃  ┗━ hello_fuchsia             ...
┣━ src/                         (if using cargo)
┃  ┗━ main.rs                   ...
┗━ target/                      ...
   ┗━ x86_64-unknown-fuchsia/   ...
      ┗━ debug/                 ...
         ┗━ hello_fuchsia       ...
```

With our Rust binary built, we can move to creating a Fuchsia package.
On Fuchsia, a package is the unit of distribution for software. We'll need to
create a new package directory where we will place files like our finished
binary and any data it may need.

To start, make the `pkg`, and `pkg/meta` directories:

```sh
mkdir pkg
mkdir pkg/meta
```

**Current directory structure**
```txt
hello_fuchsia/
┗━ pkg/
   ┗━ meta/
```

Now, create the following files inside:

**`pkg/meta/package`**
```json
{
  "name": "hello_fuchsia",
  "version": "0"
}
```

The `package` file describes our package's name and version number. Every
package must contain one.

**`pkg/hello_fuchsia.manifest` if using cargo**
```txt
bin/hello_fuchsia=target/x86_64-unknown-fuchsia/debug/hello_fuchsia
lib/ld.so.1=<SDK_PATH>/arch/x64/sysroot/dist/lib/ld.so.1
lib/libfdio.so=<SDK_PATH>/arch/x64/dist/libfdio.so
meta/package=pkg/meta/package
meta/hello_fuchsia.cm=pkg/meta/hello_fuchsia.cm
```

**`pkg/hello_fuchsia.manifest` if using rustc**
```txt
bin/hello_fuchsia=bin/hello_fuchsia
lib/ld.so.1=<SDK_PATH>/arch/x64/sysroot/dist/lib/ld.so.1
lib/libfdio.so=<SDK_PATH>/arch/x64/dist/libfdio.so
meta/package=pkg/meta/package
meta/hello_fuchsia.cm=pkg/meta/hello_fuchsia.cm
```

*Note: Relative manifest paths are resolved starting from the working directory
of `ffx`. Make sure to fill out `<SDK_PATH>` with the path to the downloaded
SDK.*

The `.manifest` file will be used to describe the contents of the package by
relating their location when installed to their location on the file system. The
`bin/hello_fuchsia=` entry will be different depending on how your Rust binary
was built, so choose accordingly.

**Current directory structure**
```txt
hello_fuchsia/
┗━ pkg/
   ┣━ meta/
   ┃  ┗━ package
   ┗━ hello_fuchsia.manifest
```

### Creating a Fuchsia component

On Fuchsia, components require a component manifest written in Fuchsia's markup
language called CML. The Fuchsia devsite contains an [overview of CML] and a
[reference for the file format]. Here's a basic one that can run our single binary:

**`pkg/hello_fuchsia.cml`**
```txt
{
    include: [ "syslog/client.shard.cml" ],
    program: {
        runner: "elf",
        binary: "bin/hello_fuchsia",
    },
}
```

**Current directory structure**
```txt
hello_fuchsia/
┗━ pkg/
   ┣━ meta/
   ┃  ┗━ package
   ┣━ hello_fuchsia.manifest
   ┗━ hello_fuchsia.cml
```

Now we can compile that CML into a component manifest:

```sh
${SDK_PATH}/tools/${ARCH}/cmc compile \
    pkg/hello_fuchsia.cml \
    --includepath ${SDK_PATH}/pkg \
    -o pkg/meta/hello_fuchsia.cm
```

*Note: `--includepath` tells the compiler where to look for `include`s from our CML.
In our case, we're only using `syslog/client.shard.cml`.*

**Current directory structure**
```txt
hello_fuchsia/
┗━ pkg/
   ┣━ meta/
   ┃  ┣━ package
   ┃  ┗━ hello_fuchsia.cm
   ┣━ hello_fuchsia.manifest
   ┗━ hello_fuchsia.cml
```

### Building a Fuchsia package

Next, we'll build a package manifest as defined by our manifest:

```sh
${SDK_PATH}/tools/${ARCH}/ffx package build \
    --api-level $(${SDK_PATH}/tools/${ARCH}/ffx --machine json version | jq .tool_version.api_level) \
    --out pkg/hello_fuchsia_manifest \
    pkg/hello_fuchsia.manifest
```

This will produce `pkg/hello_fuchsia_manifest/` which is a package manifest we can
publish directly to a repository.

**Current directory structure**
```txt
hello_fuchsia/
┗━ pkg/
   ┣━ meta/
   ┃  ┣━ package
   ┃  ┗━ hello_fuchsia.cm
   ┣━ hello_fuchsia_manifest/
   ┃  ┗━ ...
   ┣━ hello_fuchsia.manifest
   ┣━ hello_fuchsia.cml
   ┗━ hello_fuchsia_package_manifest
```

We are now ready to publish the package.

## Publishing a Fuchsia package

With our package and component manifests setup,
we can now publish our package. The first step will
be to create a Fuchsia package repository to publish
to.

### Creating a Fuchsia package repository

We can set up our repository with:

```sh
${SDK_PATH}/tools/${ARCH}/ffx repository create pkg/repo
```

**Current directory structure**
```txt
hello_fuchsia/
┗━ pkg/
   ┣━ meta/
   ┃  ┣━ package
   ┃  ┗━ hello_fuchsia.cm
   ┣━ hello_fuchsia_manifest/
   ┃  ┗━ ...
   ┣━ repo/
   ┃  ┗━ ...
   ┣━ hello_fuchsia.manifest
   ┣━ hello_fuchsia.cml
   ┗━ hello_fuchsia_package_manifest
```

## Publishing Fuchsia package to repository

We can publish our new package to that repository with:

```sh
${SDK_PATH}/tools/${ARCH}/ffx repository publish \
    --package pkg/hello_fuchsia_package_manifest \
    pkg/repo
```

## Running a Fuchsia component on an emulator

At this point, we are ready to run our Fuchsia
component. For reference, our final directory
structure will look like:

**Final directory structure**
```txt
hello_fuchsia/
┣━ src/                         (if using rustc)
┃   ┗━ hello_fuchsia.rs         ...
┣━ bin/                         ...
┃  ┗━ hello_fuchsia             ...
┣━ src/                         (if using cargo)
┃  ┗━ main.rs                   ...
┣━ target/                      ...
┃  ┗━ x86_64-unknown-fuchsia/   ...
┃     ┗━ debug/                 ...
┃        ┗━ hello_fuchsia       ...
┗━ pkg/
   ┣━ meta/
   ┃  ┣━ package
   ┃  ┗━ hello_fuchsia.cm
   ┣━ hello_fuchsia_manifest/
   ┃  ┗━ ...
   ┣━ repo/
   ┃  ┗━ ...
   ┣━ hello_fuchsia.manifest
   ┣━ hello_fuchsia.cml
   ┗━ hello_fuchsia_package_manifest
```

### Starting the Fuchsia emulator

Start a Fuchsia emulator in a new terminal using:

```sh
${SDK_PATH}/tools/${ARCH}/ffx product-bundle get workstation_eng.qemu-${ARCH}
${SDK_PATH}/tools/${ARCH}/ffx emu start workstation_eng.qemu-${ARCH} --headless
```

### Watching emulator logs

Once the emulator is running, open a separate terminal to watch the emulator logs:

**In separate terminal**
```sh
${SDK_PATH}/tools/${ARCH}/ffx log \
    --since now
```

### Serving a Fuchsia package

Now, start a package repository server to serve our
package to the emulator:

```sh
${SDK_PATH}/tools/${ARCH}/ffx repository server start \
    --background --repository hello-fuchsia --repo-path pkg-repo
```

Once the repository server is up and running, register it with the target Fuchsia system running in the emulator:

```sh
${SDK_PATH}/tools/${ARCH}/ffx target repository register \
    --repository hello-fuchsia
```

### Running a Fuchsia component

Finally, run the component:

```sh
${SDK_PATH}/tools/${ARCH}/ffx component run \
    /core/ffx-laboratory:hello_fuchsia \
    fuchsia-pkg://hello-fuchsia/hello_fuchsia_manifest#meta/hello_fuchsia.cm
```

On reruns of the component, the `--recreate` argument may also need to be
passed.

```sh
${SDK_PATH}/tools/${ARCH}/ffx component run \
    --recreate \
    /core/ffx-laboratory:hello_fuchsia \
    fuchsia-pkg://hello-fuchsia/hello_fuchsia_manifest#meta/hello_fuchsia.cm
```

## `.gitignore` extensions

Optionally, we can create/extend our `.gitignore` file to ignore files and
directories that are not helpful to track:

```txt
pkg/repo
pkg/meta/hello_fuchsia.cm
pkg/hello_fuchsia_manifest
pkg/hello_fuchsia_package_manifest
```

## Testing

### Running unit tests

Tests can be run in the same way as a regular binary.

* If using `cargo`, you can simply pass `test --no-run`
to the `cargo` invocation and then repackage and rerun the Fuchsia package. From our previous example,
this would look like `cargo test --target x86_64-unknown-fuchsia --no-run`, and moving the executable
binary path found from the line `Executable unittests src/main.rs (target/x86_64-unknown-fuchsia/debug/deps/hello_fuchsia-<HASH>)`
into `pkg/hello_fuchsia.manifest`.

* If using the compiled `rustc`, you can simply pass `--test`
to the `rustc` invocation and then repackage and rerun the Fuchsia package.

The test harness will run the applicable unit tests.

Often when testing, you may want to pass additional command line arguments to
your binary. Additional arguments can be set in the component manifest:

**`pkg/hello_fuchsia.cml`**
```txt
{
    include: [ "syslog/client.shard.cml" ],
    program: {
        runner: "elf",
        binary: "bin/hello_fuchsia",
        args: ["it_works"],
    },
}
```

This will pass the argument `it_works` to the binary, filtering the tests to
only those tests that match the pattern. There are many more configuration
options available in CML including environment variables. More documentation is
available on the [Fuchsia devsite].

### Running the compiler test suite

The commands in this section assume that they are being run from inside your
local Rust source checkout:

```sh
cd ${RUST_SRC_PATH}
```

To run the Rust test suite on an emulated Fuchsia device, you'll also need to
download a copy of the Fuchsia SDK. The current minimum supported SDK version is
[20.20240412.3.1][minimum_supported_sdk_version].

[minimum_supported_sdk_version]: https://chrome-infra-packages.appspot.com/p/fuchsia/sdk/core/linux-amd64/+/version:20.20240412.3.1

Fuchsia's test runner interacts with the Fuchsia emulator and is located at
`src/ci/docker/scripts/fuchsia-test-runner.py`. First, add the following
variables to your existing `config-env.sh`:

```sh
# TEST_TOOLCHAIN_TMP_DIR can point anywhere, but it:
#  - must be less than 108 characters, otherwise qemu can't handle the path
#  - must be consistent across calls to this file (don't use `mktemp -d` here)
export TEST_TOOLCHAIN_TMP_DIR="/tmp/rust-tmp"

# Keep existing contents of `config-env.sh` from earlier, including SDK_PATH
```

We can then use the script to start our test environment with:

```sh
( \
    source config-env.sh &&                                                   \
    src/ci/docker/scripts/fuchsia-test-runner.py start                        \
    --rust-build ${RUST_SRC_PATH}/build                                       \
    --sdk ${SDK_PATH}                                                         \
    --target {x86_64-unknown-fuchsia|aarch64-unknown-fuchsia}                 \
    --verbose                                                                 \
)
```

Where `${RUST_SRC_PATH}/build` is the `build-dir` set in `bootstrap.toml`.

Once our environment is started, we can run our tests using `x.py` as usual. The
test runner script will run the compiled tests on an emulated Fuchsia device. To
run the full `tests/ui` test suite:

```sh
( \
    source config-env.sh &&                                                   \
    ./x.py                                                                    \
    --config bootstrap.toml                                                      \
    --stage=2                                                                 \
    test tests/ui                                                             \
    --target x86_64-unknown-fuchsia                                           \
    --run=always                                                              \
    --test-args --target-rustcflags                                           \
    --test-args -Lnative=${SDK_PATH}/arch/{x64|arm64}/sysroot/lib             \
    --test-args --target-rustcflags                                           \
    --test-args -Lnative=${SDK_PATH}/arch/{x64|arm64}/lib                     \
    --test-args --target-rustcflags                                           \
    --test-args -Clink-arg=--undefined-version                                \
    --test-args --remote-test-client                                          \
    --test-args src/ci/docker/scripts/fuchsia-test-runner.py                  \
)
```

By default, `x.py` compiles test binaries with `panic=unwind`. If you built your
Rust toolchain with `-Cpanic=abort`, you need to tell `x.py` to compile test
binaries with `panic=abort` as well:

```sh
    --test-args --target-rustcflags                                           \
    --test-args -Cpanic=abort                                                 \
    --test-args --target-rustcflags                                           \
    --test-args -Zpanic_abort_tests                                           \
```

When finished testing, the test runner can be used to stop the test environment:

```sh
src/ci/docker/scripts/fuchsia-test-runner.py stop
```

## Debugging

### `zxdb`

Debugging components running on a Fuchsia emulator can be done using the
console-mode debugger: [zxdb]. We will demonstrate attaching necessary symbol
paths to debug our `hello-fuchsia` component.

### Attaching `zxdb`

In a separate terminal, issue the following command from our `hello_fuchsia`
directory to launch `zxdb`:

**In separate terminal**
```sh
${SDK_PATH}/tools/${ARCH}/ffx debug connect -- \
    --symbol-path target/x86_64-unknown-fuchsia/debug
```

* `--symbol-path` gets required symbol paths, which are
necessary for stepping through your program.

The "[displaying source code in `zxdb`](#displaying-source-code-in-zxdb)"
section describes how you can display Rust and/or Fuchsia source code in your
debugging session.

### Using `zxdb`

Once launched, you will be presented with the window:

```sh
Connecting (use "disconnect" to cancel)...
Connected successfully.
👉 To get started, try "status" or "help".
[zxdb]
```

To attach to our program, we can run:

```sh
[zxdb] attach hello_fuchsia
```

**Expected output**
```sh
Waiting for process matching "hello_fuchsia".
Type "filter" to see the current filters.
```

Next, we can create a breakpoint at main using "b main":

```sh
[zxdb] b main
```

**Expected output**
```sh
Created Breakpoint 1 @ main
```

Finally, we can re-run the "hello_fuchsia" component from our original
terminal:

```sh
${SDK_PATH}/tools/${ARCH}/ffx component run \
    --recreate \
    fuchsia-pkg://hello-fuchsia/hello_fuchsia_manifest#meta/hello_fuchsia.cm
```

Once our component is running, our `zxdb` window will stop execution
in our main as desired:

**Expected output**
```txt
Breakpoint 1 now matching 1 addrs for main
🛑 on bp 1 hello_fuchsia::main() • main.rs:2
   1 fn main() {
 ▶ 2     println!("Hello Fuchsia!");
   3 }
   4
[zxdb]
```

`zxdb` has similar commands to other debuggers like [gdb].
To list the available commands, run "help" in the
`zxdb` window or visit [the zxdb documentation].

```sh
[zxdb] help
```

**Expected output**
```sh
Help!

  Type "help <command>" for command-specific help.

Other help topics (see "help <topic>")
...
```

### Displaying source code in `zxdb`

By default, the debugger will not be able to display
source code while debugging. For our user code, we displayed
source code by pointing our debugger to our debug binary via
the `--symbol-path` arg. To display library source code in
the debugger, you must provide paths to the source using
`--build-dir`. For example, to display the Rust and Fuchsia
source code:

```sh
${SDK_PATH}/tools/${ARCH}/ffx debug connect -- \
    --symbol-path target/x86_64-unknown-fuchsia/debug \
    --build-dir ${RUST_SRC_PATH}/rust \
    --build-dir ${FUCHSIA_SRC_PATH}/fuchsia/out/default
```

 * `--build-dir` links against source code paths, which
 are not strictly necessary for debugging, but is a nice-to-have
 for displaying source code in `zxdb`.

 Linking to a Fuchsia checkout can help with debugging Fuchsia libraries,
 such as [fdio].

### Debugging the compiler test suite

Debugging the compiler test suite requires some special configuration:

First, we have to properly configure zxdb so it will be able to find debug
symbols and source information for our test. The test runner can do this for us
with:

```sh
src/ci/docker/scripts/fuchsia-test-runner.py debug                            \
    --rust-src ${RUST_SRC_PATH}                                               \
    --fuchsia-src ${FUCHSIA_SRC_PATH}                                         \
    --test ${TEST}
```

where `${TEST}` is relative to Rust's `tests` directory (e.g. `ui/abi/...`).

This will start a zxdb session that is properly configured for the specific test
being run. All three arguments are optional, so you can omit `--fuchsia-src` if
you don't have it downloaded. Now is a good time to set any desired breakpoints,
like `b main`.

Next, we have to tell `x.py` not to optimize or strip debug symbols from our
test suite binaries. We can do this by passing some new arguments to `rustc`
through our `x.py` invocation. The full invocation is:

```sh
( \
    source config-env.sh &&                                                   \
    ./x.py                                                                    \
    --config bootstrap.toml                                                      \
    --stage=2                                                                 \
    test tests/${TEST}                                                        \
    --target x86_64-unknown-fuchsia                                           \
    --run=always                                                              \
    --test-args --target-rustcflags                                           \
    --test-args -Lnative=${SDK_PATH}/arch/{x64|arm64}/sysroot/lib             \
    --test-args --target-rustcflags                                           \
    --test-args -Lnative=${SDK_PATH}/arch/{x64|arm64}/lib                     \
    --test-args --target-rustcflags                                           \
    --test-args -Clink-arg=--undefined-version                                \
    --test-args --target-rustcflags                                           \
    --test-args -Cdebuginfo=2                                                 \
    --test-args --target-rustcflags                                           \
    --test-args -Copt-level=0                                                 \
    --test-args --target-rustcflags                                           \
    --test-args -Cstrip=none                                                  \
    --test-args --remote-test-client                                          \
    --test-args src/ci/docker/scripts/fuchsia-test-runner.py                  \
)
```

*If you built your Rust toolchain with `panic=abort`, make sure to include the
previous flags so your test binaries are also compiled with `panic=abort`.*

Upon running this command, the test suite binary will be run and zxdb will
attach and load any relevant debug symbols.

[Fuchsia team]: https://team-api.infra.rust-lang.org/v1/teams/fuchsia.json
[Fuchsia]: https://fuchsia.dev/
[source tree]: https://fuchsia.dev/fuchsia-src/get-started/learn/build
[rustup]: https://rustup.rs/
[cargo]: ../../cargo/index.html
[Fuchsia SDK]: https://chrome-infra-packages.appspot.com/p/fuchsia/sdk/core
[overview of CML]: https://fuchsia.dev/fuchsia-src/concepts/components/v2/component_manifests
[reference for the file format]: https://fuchsia.dev/reference/cml
[Fuchsia devsite]: https://fuchsia.dev/reference/cml
[not currently supported]: https://fxbug.dev/105393
[zxdb]: https://fuchsia.dev/fuchsia-src/development/debugger
[gdb]: https://www.sourceware.org/gdb/
[the zxdb documentation]: https://fuchsia.dev/fuchsia-src/development/debugger
[fdio]: https://cs.opensource.google/fuchsia/fuchsia/+/main:sdk/lib/fdio/
