# Fuchsia integration tests

[Fuchsia](https://fuchsia.dev) is an open-source operating system with about 2
million lines of Rust code.[^loc] It has caught a large number of [regressions]
in the past and was subsequently included in CI.

## What to do if the Fuchsia job breaks?

Please contact the [fuchsia][fuchsia-ping] ping group and ask them for help.

```text
@rustbot ping fuchsia
```

## Building Fuchsia in CI

Fuchsia builds as part of the suite of bors tests that run before a pull request
is merged.

If you are worried that a pull request might break the Fuchsia builder and want
to test it out before submitting it to the bors queue, simply add this line to
your PR description:

> try-job: x86_64-fuchsia

Then when you `@bors try` it will pick the job that builds Fuchsia.

## Building Fuchsia locally

Because Fuchsia uses languages other than Rust, it does not use Cargo as a build
system. It also requires the toolchain build to be configured in a [certain
way][build-toolchain].

The recommended way to build Fuchsia is to use the Docker scripts that check out
and run a Fuchsia build for you. If you've run Docker tests before, you can
simply run this command from your Rust checkout to download and build Fuchsia
using your local Rust toolchain.

```
src/ci/docker/run.sh x86_64-fuchsia
```

See the [Testing with Docker](../docker.md) chapter for more details on how to run
and debug jobs with Docker.

Note that a Fuchsia checkout is *large* – as of this writing, a checkout and
build takes 46G of space – and as you might imagine, it takes a while to
complete.

### Modifying the Fuchsia checkout

The main reason you would want to build Fuchsia locally is because you need to
investigate a regression. After running a Docker build, you'll find the Fuchsia
checkout inside the `obj/fuchsia` directory of your Rust checkout.  If you
modify the `KEEP_CHECKOUT` line in the [build-fuchsia.sh] script to
`KEEP_CHECKOUT=1`, you can change the checkout as needed and rerun the build
command above. This will reuse all the build results from before.

You can find more options to customize the Fuchsia checkout in the
[build-fuchsia.sh] script.

### Customizing the Fuchsia build

You can find more info about the options used to build Fuchsia in Rust CI in the
[build_fuchsia_from_rust_ci.sh] script invoked by [build-fuchsia.sh].

The Fuchsia build system uses [GN], a metabuild system that generates [Ninja]
files and then hands off the work of running the build to Ninja.

Fuchsia developers use `fx` to run builds and perform other development tasks.
This tool is located in `.jiri_root/bin` of the Fuchsia checkout; you may need
to add this to your `$PATH` for some workflows.

There are a few `fx` subcommands that are relevant, including:

- `fx set` accepts build arguments, writes them to `out/default/args.gn`, and
  runs GN.
- `fx build` builds the Fuchsia project using Ninja. It will automatically pick
  up changes to build arguments and rerun GN. By default it builds everything,
  but it also accepts target paths to build specific targets (see below).
- `fx clippy` runs Clippy on specific Rust targets (or all of them). We use this
  in the Rust CI build to avoid running codegen on most Rust targets. Underneath
  it invokes Ninja, just like `fx build`. The clippy results are saved in json
  files inside the build output directory before being printed.

#### Target paths

GN uses paths like the following to identify build targets:

```
//src/starnix/kernel:starnix_core
```

The initial `//` means the root of the checkout, and the remaining slashes are
directory names. The string after `:` is the _target name_ of a target defined
in the `BUILD.gn` file of that directory.

The target name can be omitted if it is the same as the directory name. In other
words, `//src/starnix/kernel` is the same as `//src/starnix/kernel:kernel`.

These target paths are used inside `BUILD.gn` files to reference dependencies,
and can also be used in `fx build`.

#### Modifying compiler flags

You can put custom compiler flags inside a GN `config` that is added to a
target. As a simple example:

```
config("everybody_loops") {
    rustflags = [ "-Zeverybody-loops" ]
}

rustc_binary("example") {
    crate_root = "src/bin.rs"
    # ...existing keys here...
    configs += [ ":everybody_loops" ]
}
```

This will add the flag `-Zeverybody-loops` to rustc when building the `example`
target. Note that you can also use [`public_configs`] for a config to be added
to every target that depends on that target.

If you want to add a flag to every Rust target in the build, you can add
rustflags to the [`//build/config:compiler`] config or to the OS-specific
configs referenced in that file. Note that `cflags` and `ldflags` are ignored on
Rust targets.

#### Running ninja and rustc commands directly

Going down one layer, `fx build` invokes `ninja`, which in turn eventually
invokes `rustc`. All build actions are run inside the out directory, which is
usually `out/default` inside the Fuchsia checkout.

You can get ninja to print the actual command it invokes by forcing that command
to fail, e.g. by adding a syntax error to one of the source files of the target.
Once you have the command, you can run it from inside the output directory.

After changing the toolchain itself, the build setting `rustc_version_string` in
`out/default/args.gn` needs to be changed so that `fx build` or `ninja` will
rebuild all the Rust targets. This can be done in a text editor and the contents
of the string do not matter, as long as it changes from one build to the next.
[build_fuchsia_from_rust_ci.sh] does this for you by hashing the toolchain
directory.

The Fuchsia website has more detailed documentation of the [build system].

#### Other tips and tricks

When using `build_fuchsia_from_rust_ci.sh` you can comment out the `fx set`
command after the initial run so it won't rerun GN each time. If you do this you
can also comment out the version_string line to save a couple seconds.

`export NINJA_PERSISTENT_MODE=1` to get faster ninja startup times after the
initial build.

## Fuchsia target support

To learn more about Fuchsia target support, see the Fuchsia chapter in [the
rustc book][platform-support].

[regressions]: https://gist.github.com/tmandry/7103eba4bd6a6fb0c439b5a90ae355fa
[build-toolchain]: https://fuchsia.dev/fuchsia-src/development/build/rust_toolchain
[build-fuchsia.sh]: https://github.com/rust-lang/rust/blob/221e2741c39515a5de6da42d8c76ee1e132c2c74/src/ci/docker/host-x86_64/x86_64-fuchsia/build-fuchsia.sh
[build_fuchsia_from_rust_ci.sh]: https://cs.opensource.google/fuchsia/fuchsia/+/main:scripts/rust/build_fuchsia_from_rust_ci.sh?q=build_fuchsia_from_rust_ci&ss=fuchsia
[platform-support]: https://doc.rust-lang.org/nightly/rustc/platform-support/fuchsia.html
[GN]: https://gn.googlesource.com/gn/+/main#gn
[Ninja]: https://ninja-build.org/
[`public_configs`]: https://gn.googlesource.com/gn/+/main/docs/reference.md#var_public_configs
[`//build/config:compiler`]: https://cs.opensource.google/fuchsia/fuchsia/+/main:build/config/BUILD.gn;l=121;drc=c26c473bef93b33117ae417893118907a026fec7
[build system]: https://fuchsia.dev/fuchsia-src/development/build/build_system
[fuchsia-ping]: ../../notification-groups/fuchsia.md

[^loc]: As of June 2024, Fuchsia had about 2 million lines of first-party Rust
code and a roughly equal amount of third-party code, as counted by tokei
(excluding comments and blanks).
