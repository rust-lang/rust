# `aarch64-fuchsia` and `x86_64-fuchsia`

**Tier: 2**

[Fuchsia] is a modern open source operating system that's simple, secure,
updatable, and performant.

## Target maintainers

The [Fuchsia team]:

- Tyler Mandry ([@tmandry](https://github.com/tmandry))
- Dan Johnson ([@computerdruid](https://github.com/computerdruid))
- David Koloski ([@djkoloski](https://github.com/djkoloski))
- Andrew Pollack ([@andrewpollack](https://github.com/andrewpollack))
- Joseph Ryan ([@P1n3appl3](https://github.com/P1n3appl3))

As the team evolves over time, the specific members listed here may differ from
the members reported by the API. The API should be considered to be
authoritative if this occurs. Instead of pinging individual members, use
`@rustbot ping fuchsia` to contact the team on GitHub.

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

For the purposes of this walkthrough, we will only target `x86_64-fuchsia`.

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
rustup target add x86_64-fuchsia
rustup target add aarch64-fuchsia
```

After installing our Fuchsia targets, we can now compile a Rust binary that targets
Fuchsia.

To create our Rust project, we can issue a standard `cargo` command as follows:

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
[target.x86_64-fuchsia]

rustflags = [
    "-Lnative=<SDK_PATH>/arch/x64/lib",
    "-Lnative=<SDK_PATH>/arch/x64/sysroot/lib"
]
```

*Note: Make sure to fill out `<SDK_PATH>` with the path to the downloaded [Fuchsia SDK].*

These options configure the following:

* `-Lnative=${SDK_PATH}/arch/${ARCH}/lib`: Link against Fuchsia libraries from
  the SDK
* `-Lnative=${SDK_PATH}/arch/${ARCH}/sysroot/lib`: Link against Fuchsia kernel
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
cargo build --target x86_64-fuchsia
```

Now we have a Rust binary at `target/x86_64-fuchsia/debug/hello_fuchsia`,
targeting our desired Fuchsia target.

**Current directory structure**
```txt
hello_fuchsia/
┣━ src/
┃  ┗━ main.rs
┣━ target/
┃  ┗━ x86_64-fuchsia/
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
configuration.

In `config.toml`, add:

```toml
[build]
target = ["<host_platform>", "aarch64-fuchsia", "x86_64-fuchsia"]
```

Additionally, the following environment variables must be configured (for
example, using a script like `config-env.sh`):

```sh
# Configure this environment variable to be the path to the downloaded SDK
export SDK_PATH="<SDK path goes here>"

export CFLAGS_aarch64_fuchsia="--target=aarch64-fuchsia --sysroot=${SDK_PATH}/arch/arm64/sysroot -I${SDK_PATH}/pkg/fdio/include"
export CXXFLAGS_aarch64_fuchsia="--target=aarch64-fuchsia --sysroot=${SDK_PATH}/arch/arm64/sysroot -I${SDK_PATH}/pkg/fdio/include"
export LDFLAGS_aarch64_fuchsia="--target=aarch64-fuchsia --sysroot=${SDK_PATH}/arch/arm64/sysroot -L${SDK_PATH}/arch/arm64/lib"
export CARGO_TARGET_AARCH64_FUCHSIA_RUSTFLAGS="-C link-arg=--sysroot=${SDK_PATH}/arch/arm64/sysroot -Lnative=${SDK_PATH}/arch/arm64/sysroot/lib -Lnative=${SDK_PATH}/arch/arm64/lib"
export CFLAGS_x86_64_fuchsia="--target=x86_64-fuchsia --sysroot=${SDK_PATH}/arch/x64/sysroot -I${SDK_PATH}/pkg/fdio/include"
export CXXFLAGS_x86_64_fuchsia="--target=x86_64-fuchsia --sysroot=${SDK_PATH}/arch/x64/sysroot -I${SDK_PATH}/pkg/fdio/include"
export LDFLAGS_x86_64_fuchsia="--target=x86_64-fuchsia --sysroot=${SDK_PATH}/arch/x64/sysroot -L${SDK_PATH}/arch/x64/lib"
export CARGO_TARGET_X86_64_FUCHSIA_RUSTFLAGS="-C link-arg=--sysroot=${SDK_PATH}/arch/x64/sysroot -Lnative=${SDK_PATH}/arch/x64/sysroot/lib -Lnative=${SDK_PATH}/arch/x64/lib"
```

These can be run together in a shell environment by executing
`(source config-env.sh && ./x.py install)`.

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

* `--target x86_64-fuchsia`/`--target aarch64-fuchsia`: Targets the Fuchsia
  platform of your choice
* `-Lnative ${SDK_PATH}/arch/${ARCH}/lib`: Link against Fuchsia libraries from
  the SDK
* `-Lnative ${SDK_PATH}/arch/${ARCH}/sysroot/lib`: Link against Fuchsia kernel
  libraries from the SDK

Putting it all together:

```sh
# Configure these for the Fuchsia target of your choice
TARGET_ARCH="<x86_64-fuchsia|aarch64-fuchsia>"
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
┣━ src/                     (if using rustc)
┃   ┗━ hello_fuchsia.rs     ...
┣━ bin/                     ...
┃  ┗━ hello_fuchsia         ...
┣━ src/                     (if using cargo)
┃  ┗━ main.rs               ...
┗━ target/                  ...
   ┗━ x86_64-fuchsia/       ...
      ┗━ debug/             ...
         ┗━ hello_fuchsia   ...
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

**`pkg/hello_fuchsia.manifest`**
```txt
bin/hello_fuchsia=target/x86_64-fuchsia/debug/hello_fuchsia     # If using cargo...
bin/hello_fuchsia=bin/hello_fuchsia                             # If using rustc...
lib/ld.so.1=<SDK_PATH>/arch/x64/sysroot/dist/lib/ld.so.1
lib/libfdio.so=<SDK_PATH>/arch/x64/dist/libfdio.so
meta/package=pkg/meta/package
meta/hello_fuchsia.cm=pkg/meta/hello_fuchsia.cm
```

*Note: Relative manifest paths are resolved starting from the working directory
of `pm`. Make sure to fill out `<SDK_PATH>` with the path to the downloaded
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
${SDK_PATH}/tools/${ARCH}/pm \
    -o pkg/hello_fuchsia_manifest \
    -m pkg/hello_fuchsia.manifest \
    build \
    -output-package-manifest pkg/hello_fuchsia_package_manifest
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
${SDK_PATH}/tools/${ARCH}/pm newrepo \
    -repo pkg/repo
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
${SDK_PATH}/tools/${ARCH}/pm publish \
    -repo pkg/repo \
    -lp -f <(echo "pkg/hello_fuchsia_package_manifest")
```

Then we can add the repository to `ffx`'s package server as `hello-fuchsia` using:

```sh
${SDK_PATH}/tools/${ARCH}/ffx repository add-from-pm \
    pkg/repo \
    -r hello-fuchsia
```

## Running a Fuchsia component on an emulator

At this point, we are ready to run our Fuchsia
component. For reference, our final directory
structure will look like:

**Final directory structure**
```txt
hello_fuchsia/
┣━ src/                     (if using rustc)
┃   ┗━ hello_fuchsia.rs     ...
┣━ bin/                     ...
┃  ┗━ hello_fuchsia         ...
┣━ src/                     (if using cargo)
┃  ┗━ main.rs               ...
┣━ target/                  ...
┃  ┗━ x86_64-fuchsia/       ...
┃     ┗━ debug/             ...
┃        ┗━ hello_fuchsia   ...
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
${SDK_PATH}/tools/${ARCH}/ffx repository server start
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
    fuchsia-pkg://hello-fuchsia/hello_fuchsia_manifest#meta/hello_fuchsia.cm
```

On reruns of the component, the `--recreate` argument may also need to be
passed.

```sh
${SDK_PATH}/tools/${ARCH}/ffx component run \
    --recreate \
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
this would look like `cargo test --target x86_64-fuchsia --no-run`, and moving the executable
binary path found from the line `Executable unittests src/main.rs (target/x86_64-fuchsia/debug/deps/hello_fuchsia-<HASH>)`
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

Running the Rust test suite on Fuchsia is [not currently supported], but work is
underway to enable it.

[Fuchsia team]: https://team-api.infra.rust-lang.org/v1/teams/fuchsia.json
[Fuchsia]: https://fuchsia.dev/
[source tree]: https://fuchsia.dev/fuchsia-src/get-started/learn/build
[rustup]: https://rustup.rs/
[cargo]: https://doc.rust-lang.org/cargo/
[Fuchsia SDK]: https://chrome-infra-packages.appspot.com/p/fuchsia/sdk/core
[overview of CML]: https://fuchsia.dev/fuchsia-src/concepts/components/v2/component_manifests
[reference for the file format]: https://fuchsia.dev/reference/cml
[Fuchsia devsite]: https://fuchsia.dev/reference/cml
[not currently supported]: https://fxbug.dev/105393
