# `aarch64-fuchsia` and `x86_64-fuchsia`

**Tier: 2**

[Fuchsia] is a modern open source operating system that's simple, secure,
updatable, and performant.

[Fuchsia]: https://fuchsia.dev/

## Target maintainers

The [Fuchsia team]:

[Fuchsia team]: https://team-api.infra.rust-lang.org/v1/teams/fuchsia.json

- Tyler Mandry ([@tmandry](https://github.com/tmandry))
- Dan Johnson ([@computerdruid](https://github.com/computerdruid))
- David Koloski ([@djkoloski](https://github.com/djkoloski))
- Andrew Pollack ([@andrewpollack](https://github.com/andrewpollack))
- Joseph Ryan ([@P1n3appl3](https://github.com/P1n3appl3))

As the team evolves over time, the specific members listed here may differ from
the members reported by the API. The API should be considered to be
authoritative if this occurs. Instead of pinging individual members, use
`@rustbot ping fuchsia` to contact the team on GitHub.

## Requirements

This target is cross-compiled from a host environment. Development may be done
from the [source tree] or using the Fuchsia SDK.

[source tree]: https://fuchsia.dev/fuchsia-src/get-started/learn/build

Fuchsia targets support std and follow the `sysv64` calling convention on
x86_64. Fuchsia binaries use the ELF file format.

## Building the target

Before building Rust for Fuchsia, you'll need a clang toolchain that supports
Fuchsia as well. A recent version (14+) of clang should be sufficient to compile
Rust for Fuchsia.

You'll also need a recent copy of the [Fuchsia SDK], which provides the tools
and binaries required to build and link programs for Fuchsia.

[Fuchsia SDK]: https://chrome-infra-packages.appspot.com/p/fuchsia/sdk/core

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

## Building Rust programs

After compiling Rust binaries, you'll need to build a component, package it, and
serve it to a Fuchsia device or emulator. All of this can be done using the
Fuchsia SDK.

As an example, we'll compile and run this simple program on a Fuchsia emulator:

**`hello_fuchsia.rs`**
```rust
fn main() {
    println!("Hello Fuchsia!");
}

#[test]
fn it_works() {
    assert_eq!(2 + 2, 4);
}
```

Create a new file named `hello_fuchsia.rs` and fill out its contents with that
code.

### Create a package

On Fuchsia, a package is the unit of distribution for software. We'll need to
create a new package directory where we will place files like our finished
binary and any data it may need. The working directory will have this layout:

```txt
hello_fuchsia.rs
hello_fuchsia.cml
package
┣━ bin
┃  ┗━ hello_fuchsia
┣━ meta
┃  ┣━ package
┃  ┗━ hello_fuchsia.cm
┗━ hello_fuchsia.manifest
```

Make the `package`, `package/bin`, and `package/meta` directories and create the
following files inside:

**`package/meta/package`**
```json
{
  "name": "hello_fuchsia",
  "version": "0"
}
```

The `package` file describes our package's name and version number. Every
package must contain one.

**`package/hello_fuchsia.manifest`**
```txt
bin/hello_fuchsia=package/bin/hello_fuchsia
lib/ld.so.1=<SDK_PATH>/arch/x64/sysroot/dist/lib/ld.so.1
lib/libfdio.so=<SDK_PATH>/arch/x64/dist/libfdio.so
meta/package=package/meta/package
meta/hello_fuchsia.cm=package/meta/hello_fuchsia.cm
```

*Note: Relative manifest paths are resolved starting from the working directory
of `pm`. Make sure to fill out `<SDK_PATH>` with the path to the downloaded
SDK.*

The `.manifest` file will be used to describe the contents of the package by
relating their location when installed to their location on the file system. You
can use this to make a package pull files from other places, but for this
example we'll just be placing everything in the `package` directory.

### Compiling a binary

Using your freshly compiled `rustc`, you can compile a binary for Fuchsia using
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

rustc --target ${TARGET_ARCH} -Lnative=${SDK_PATH}/arch/${ARCH}/lib -Lnative=${SDK_PATH}/arch/${ARCH}/sysroot/lib -o package/bin/hello_fuchsia hello_fuchsia.rs
```

### Bulding a component

On Fuchsia, components require a component manifest written in Fuchia's markup
language called CML. The Fuchsia devsite contains an [overview of CML] and a
[reference for the file format]. Here's a basic one that can run our single binary:

[overview of CML]: https://fuchsia.dev/fuchsia-src/concepts/components/v2/component_manifests
[reference for the file format]: https://fuchsia.dev/reference/cml

**`hello_fuchsia.cml`**
```txt
{
    include: [ "syslog/client.shard.cml" ],
    program: {
        runner: "elf",
        binary: "bin/hello_fuchsia",
    },
}
```

Now we can compile that CML into a component manifest:

```sh
${SDK_PATH}/tools/${ARCH}/cmc compile hello_fuchsia.cml --includepath ${SDK_PATH}/pkg -o package/meta/hello_fuchsia.cm
```

`--includepath` tells the compiler where to look for `include`s from our CML.
In our case, we're only using `syslog/client.shard.cml`.

### Building and publishing a package

Next, we'll build our package as defined by our manifest:

```sh
${SDK_PATH}/tools/${ARCH}/pm -o hello_fuchsia -m package/hello_fuchsia.manifest build -output-package-manifest hello_fuchsia_manifest
```

This will produce `hello_fuchsia_manifest` which is a package manifest we can
publish directly to a repository. We can set up that repository with:

```sh
${SDK_PATH}/tools/${ARCH}/pm newrepo -repo repo
```

And then publish our new package to that repository with:

```sh
${SDK_PATH}/tools/${ARCH}/pm publish -repo repo -lp -f <(echo "hello_fuchsia_manifest")
```

Then we can add it to `ffx`'s package server as `hello-fuchsia` using:

```sh
${SDK_PATH}/tools/${ARCH}/ffx repository add-from-pm repo -r hello-fuchsia
```

### Starting the emulator

Start a Fuchsia emulator in a new terminal using:

```sh
${SDK_PATH}/tools/${ARCH}/ffx product-bundle get workstation_eng.qemu-${ARCH}
${SDK_PATH}/tools/${ARCH}/ffx emu start workstation_eng.qemu-${ARCH} --headless
```

Once the emulator is running, start a package repository server to serve our
package to the emulator:

```sh
${SDK_PATH}/tools/${ARCH}/ffx repository server start
```

Once the repository server is up and running, register our repository:

```sh
${SDK_PATH}/tools/${ARCH}/ffx target repository register --repository hello-fuchsia
```

And watch the logs from the emulator in a separate terminal:

```sh
${SDK_PATH}/tools/${ARCH}/ffx log --since now
```

Finally, run the component:

```sh
${SDK_PATH}/tools/${ARCH}/ffx component run fuchsia-pkg://hello-fuchsia/hello_fuchsia#meta/hello_fuchsia.cm
```

On reruns of the component, the `--recreate` argument may also need to be
passed.

```sh
${SDK_PATH}/tools/${ARCH}/ffx component run --recreate fuchsia-pkg://hello-fuchsia/hello_fuchsia#meta/hello_fuchsia.cm
```

## Testing

### Running unit tests

Tests can be run in the same way as a regular binary, simply by passing `--test`
to the `rustc` invocation and then repackaging and rerunning. The test harness
will run the applicable unit tests.

Often when testing, you may want to pass additional command line arguments to
your binary. Additional arguments can be set in the component manifest:

**`hello_fuchsia.cml`**
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
available on the [Fuchsia devsite](https://fuchsia.dev/reference/cml).

### Running the compiler test suite

Running the Rust test suite on Fuchsia is [not currently supported], but work is
underway to enable it.

[not currently supported]: https://fxbug.dev/105393
