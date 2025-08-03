# How to build and run the compiler

<div class="warning">

For `profile = "library"` users, or users who use `download-rustc = true | "if-unchanged"`, please be advised that
the `./x test library/std` flow where `download-rustc` is active (i.e. no compiler changes) is currently broken.
This is tracked in <https://github.com/rust-lang/rust/issues/142505>. Only the `./x test` flow is affected in this
case, `./x {check,build} library/std` should still work.

In the short-term, you may need to disable `download-rustc` for `./x test library/std`. This can be done either by:

1. `./x test library/std --set rust.download-rustc=false`
2. Or set `rust.download-rustc=false` in `bootstrap.toml`.

Unfortunately that will require building the stage 1 compiler. The bootstrap team is working on this, but
implementing a maintainable fix is taking some time.

</div>


The compiler is built using a tool called `x.py`. You will need to
have Python installed to run it.

## Quick Start

For a less in-depth quick-start of getting the compiler running, see [quickstart](./quickstart.md).


## Get the source code

The main repository is [`rust-lang/rust`][repo]. This contains the compiler,
the standard library (including `core`, `alloc`, `test`, `proc_macro`, etc),
and a bunch of tools (e.g. `rustdoc`, the bootstrapping infrastructure, etc).

[repo]: https://github.com/rust-lang/rust

The very first step to work on `rustc` is to clone the repository:

```bash
git clone https://github.com/rust-lang/rust.git
cd rust
```

### Partial clone the repository

Due to the size of the repository, cloning on a slower internet connection can take a long time,
and requires disk space to store the full history of every file and directory.
Instead, it is possible to tell git to perform a _partial clone_, which will only fully retrieve
the current file contents, but will automatically retrieve further file contents when you, e.g.,
jump back in the history.
All git commands will continue to work as usual, at the price of requiring an internet connection
to visit not-yet-loaded points in history.

```bash
git clone --filter='blob:none' https://github.com/rust-lang/rust.git
cd rust
```

> **NOTE**: [This link](https://github.blog/open-source/git/get-up-to-speed-with-partial-clone-and-shallow-clone/)
> describes this type of checkout in more detail, and also compares it to other modes, such as
> shallow cloning.

### Shallow clone the repository

An older alternative to partial clones is to use shallow clone the repository instead.
To do so, you can use the `--depth N` option with the `git clone` command.
This instructs `git` to perform a "shallow clone", cloning the repository but truncating it to
the last `N` commits.

Passing `--depth 1` tells `git` to clone the repository but truncate the history to the latest
commit that is on the `master` branch, which is usually fine for browsing the source code or
building the compiler.

```bash
git clone --depth 1 https://github.com/rust-lang/rust.git
cd rust
```

> **NOTE**: A shallow clone limits which `git` commands can be run.
> If you intend to work on and contribute to the compiler, it is
> generally recommended to fully clone the repository [as shown above](#get-the-source-code),
> or to perform a [partial clone](#partial-clone-the-repository) instead.
>
> For example, `git bisect` and `git blame` require access to the commit history,
> so they don't work if the repository was cloned with `--depth 1`.

## What is `x.py`?

`x.py` is the build tool for the `rust` repository. It can build docs, run tests, and compile the
compiler and standard library.

This chapter focuses on the basics to be productive, but
if you want to learn more about `x.py`, [read this chapter][bootstrap].

[bootstrap]: ./bootstrapping/intro.md

Also, using `x` rather than `x.py` is recommended as:

> `./x` is the most likely to work on every system (on Unix it runs the shell script
> that does python version detection, on Windows it will probably run the
> powershell script - certainly less likely to break than `./x.py` which often just
> opens the file in an editor).[^1]

(You can find the platform related scripts around the `x.py`, like `x.ps1`)

Notice that this is not absolute. For instance, using Nushell in VSCode on Win10,
typing `x` or `./x` still opens `x.py` in an editor rather than invoking the program. :)

In the rest of this guide, we use `x` rather than `x.py` directly. The following
command:

```bash
./x check
```

could be replaced by:

```bash
./x.py check
```

### Running `x.py`

The `x.py` command can be run directly on most Unix systems in the following format:

```sh
./x <subcommand> [flags]
```

This is how the documentation and examples assume you are running `x.py`.
Some alternative ways are:

```sh
# On a Unix shell if you don't have the necessary `python3` command
./x <subcommand> [flags]

# In Windows Powershell (if powershell is configured to run scripts)
./x <subcommand> [flags]
./x.ps1 <subcommand> [flags]

# On the Windows Command Prompt (if .py files are configured to run Python)
x.py <subcommand> [flags]

# You can also run Python yourself, e.g.:
python x.py <subcommand> [flags]
```

On Windows, the Powershell commands may give you an error that looks like this:
```
PS C:\Users\vboxuser\rust> ./x
./x : File C:\Users\vboxuser\rust\x.ps1 cannot be loaded because running scripts is disabled on this system. For more
information, see about_Execution_Policies at https:/go.microsoft.com/fwlink/?LinkID=135170.
At line:1 char:1
+ ./x
+ ~~~
    + CategoryInfo          : SecurityError: (:) [], PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
```

You can avoid this error by allowing powershell to run local scripts:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Running `x.py` slightly more conveniently

There is a binary that wraps `x.py` called `x` in `src/tools/x`. All it does is
run `x.py`, but it can be installed system-wide and run from any subdirectory
of a checkout. It also looks up the appropriate version of `python` to use.

You can install it with `cargo install --path src/tools/x`.

To clarify that this is another global installed binary util, which is
similar to the one declared in section [What is `x.py`](#what-is-xpy), but
it works as an independent process to execute the `x.py` rather than calling the 
shell to run the platform related scripts.

## Create a `bootstrap.toml`

To start, run `./x setup` and select the `compiler` defaults. This will do some initialization
and create a `bootstrap.toml` for you with reasonable defaults. If you use a different default (which
you'll likely want to do if you want to contribute to an area of rust other than the compiler, such
as rustdoc), make sure to read information about that default (located in `src/bootstrap/defaults`)
as the build process may be different for other defaults.

Alternatively, you can write `bootstrap.toml` by hand. See `bootstrap.example.toml` for all the available
settings and explanations of them. See `src/bootstrap/defaults` for common settings to change.

If you have already built `rustc` and you change settings related to LLVM, then you may have to
execute `rm -rf build` for subsequent configuration changes to take effect. Note that `./x
clean` will not cause a rebuild of LLVM.

## Common `x` commands

Here are the basic invocations of the `x` commands most commonly used when
working on `rustc`, `std`, `rustdoc`, and other tools.

| Command     | When to use it                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| `./x check` | Quick check to see if most things compile; [rust-analyzer can run this automatically for you][rust-analyzer] |
| `./x build` | Builds `rustc`, `std`, and `rustdoc`                                                                         |
| `./x test`  | Runs all tests                                                                                               |
| `./x fmt`   | Formats all code                                                                                             |

As written, these commands are reasonable starting points. However, there are
additional options and arguments for each of them that are worth learning for
serious development work. In particular, `./x build` and `./x test`
provide many ways to compile or test a subset of the code, which can save a lot
of time.

Also, note that `x` supports all kinds of path suffixes for `compiler`, `library`,
and `src/tools` directories. So, you can simply run `x test tidy` instead of
`x test src/tools/tidy`. Or, `x build std` instead of `x build library/std`.

[rust-analyzer]: suggested.html#configuring-rust-analyzer-for-rustc

See the chapters on
[testing](../tests/running.md) and [rustdoc](../rustdoc.md) for more details.

### Building the compiler

Note that building will require a relatively large amount of storage space.
You may want to have upwards of 10 or 15 gigabytes available to build the compiler.

Once you've created a `bootstrap.toml`, you are now ready to run
`x`. There are a lot of options here, but let's start with what is
probably the best "go to" command for building a local compiler:

```bash
./x build library
```

This may *look* like it only builds the standard library, but that is not the case.
What this command does is the following:

- Build `rustc` using the stage0 compiler
  - This produces the stage1 compiler
- Build `std` using the stage1 compiler

This final product (stage1 compiler + libs built using that compiler)
is what you need to build other Rust programs (unless you use `#![no_std]` or
`#![no_core]`).

You will probably find that building the stage1 `std` is a bottleneck for you,
but fear not, there is a (hacky) workaround...
see [the section on avoiding rebuilds for std][keep-stage].

[keep-stage]: ./suggested.md#faster-builds-with---keep-stage

Sometimes you don't need a full build. When doing some kind of
"type-based refactoring", like renaming a method, or changing the
signature of some function, you can use `./x check` instead for a much faster build.

Note that this whole command just gives you a subset of the full `rustc`
build. The **full** `rustc` build (what you get with `./x build
--stage 2 compiler/rustc`) has quite a few more steps:

- Build `rustc` with the stage1 compiler.
  - The resulting compiler here is called the "stage2" compiler, which uses stage1 std from the previous command.
- Build `librustdoc` and a bunch of other things with the stage2 compiler.

You almost never need to do this.

### Build specific components

If you are working on the standard library, you probably don't need to build
every other default component. Instead, you can build a specific component by
providing its name, like this:

```bash
./x build --stage 1 library
```

If you choose the `library` profile when running `x setup`, you can omit `--stage 1` (it's the
default).

## Creating a rustup toolchain

Once you have successfully built `rustc`, you will have created a bunch
of files in your `build` directory. In order to actually run the
resulting `rustc`, we recommend creating rustup toolchains. The first
one will run the stage1 compiler (which we built above). The second
will execute the stage2 compiler (which we did not build, but which
you will likely need to build at some point; for example, if you want
to run the entire test suite).

```bash
rustup toolchain link stage1 build/host/stage1
rustup toolchain link stage2 build/host/stage2
```

Now you can run the `rustc` you built with. If you run with `-vV`, you
should see a version number ending in `-dev`, indicating a build from
your local environment:

```bash
$ rustc +stage1 -vV
rustc 1.48.0-dev
binary: rustc
commit-hash: unknown
commit-date: unknown
host: x86_64-unknown-linux-gnu
release: 1.48.0-dev
LLVM version: 11.0
```

The rustup toolchain points to the specified toolchain compiled in your `build` directory,
so the rustup toolchain will be updated whenever `x build` or `x test` are run for
that toolchain/stage.

**Note:** the toolchain we've built does not include `cargo`.  In this case, `rustup` will
fall back to using `cargo` from the installed `nightly`, `beta`, or `stable` toolchain
(in that order).  If you need to use unstable `cargo` flags, be sure to run
`rustup install nightly` if you haven't already.  See the
[rustup documentation on custom toolchains](https://rust-lang.github.io/rustup/concepts/toolchains.html#custom-toolchains).

**Note:** rust-analyzer and IntelliJ Rust plugin use a component called
`rust-analyzer-proc-macro-srv` to work with proc macros. If you intend to use a
custom toolchain for a project (e.g. via `rustup override set stage1`) you may
want to build this component:

```bash
./x build proc-macro-srv-cli
```

## Building targets for cross-compilation

To produce a compiler that can cross-compile for other targets,
pass any number of `target` flags to `x build`.
For example, if your host platform is `x86_64-unknown-linux-gnu`
and your cross-compilation target is `wasm32-wasip1`, you can build with:

```bash
./x build --target x86_64-unknown-linux-gnu,wasm32-wasip1
```

Note that if you want the resulting compiler to be able to build crates that
involve proc macros or build scripts, you must be sure to explicitly build target support for the
host platform (in this case, `x86_64-unknown-linux-gnu`).

If you want to always build for other targets without needing to pass flags to `x build`,
you can configure this in the `[build]` section of your `bootstrap.toml` like so:

```toml
[build]
target = ["x86_64-unknown-linux-gnu", "wasm32-wasip1"]
```

Note that building for some targets requires having external dependencies installed
(e.g. building musl targets requires a local copy of musl).
Any target-specific configuration (e.g. the path to a local copy of musl)
will need to be provided by your `bootstrap.toml`.
Please see `bootstrap.example.toml` for information on target-specific configuration keys.

For examples of the complete configuration necessary to build a target, please visit
[the rustc book](https://doc.rust-lang.org/rustc/platform-support.html),
select any target under the "Platform Support" heading on the left,
and see the section related to building a compiler for that target.
For targets without a corresponding page in the rustc book,
it may be useful to [inspect the Dockerfiles](../tests/docker.md)
that the Rust infrastructure itself uses to set up and configure cross-compilation.

If you have followed the directions from the prior section on creating a rustup toolchain,
then once you have built your compiler you will be able to use it to cross-compile like so:

```bash
cargo +stage1 build --target wasm32-wasip1
```

## Other `x` commands

Here are a few other useful `x` commands. We'll cover some of them in detail
in other sections:

- Building things:
  - `./x build` – builds everything using the stage 1 compiler,
    not just up to `std`
  - `./x build --stage 2` – builds everything with the stage 2 compiler including
    `rustdoc`
- Running tests (see the [section on running tests](../tests/running.html) for
  more details):
  - `./x test library/std` – runs the unit tests and integration tests from `std`
  - `./x test tests/ui` – runs the `ui` test suite
  - `./x test tests/ui/const-generics` - runs all the tests in
    the `const-generics/` subdirectory of the `ui` test suite
  - `./x test tests/ui/const-generics/const-types.rs` - runs
    the single test `const-types.rs` from the `ui` test suite

### Cleaning out build directories

Sometimes you need to start fresh, but this is normally not the case.
If you need to run this then bootstrap is most likely not acting right and
you should file a bug as to what is going wrong. If you do need to clean
everything up then you only need to run one command!

```bash
./x clean
```

`rm -rf build` works too, but then you have to rebuild LLVM, which can take
a long time even on fast computers.

## Remarks on disk space

Building the compiler (especially if beyond stage 1) can require significant amounts of free disk
space, possibly around 100GB. This is compounded if you have a separate build directory for
rust-analyzer (e.g. `build-rust-analyzer`). This is easy to hit with dev-desktops which have a [set
disk
quota](https://github.com/rust-lang/simpleinfra/blob/8a59e4faeb75a09b072671c74a7cb70160ebef50/ansible/roles/dev-desktop/defaults/main.yml#L7)
for each user, but this also applies to local development as well. Occasionally, you may need to:

- Remove `build/` directory.
- Remove `build-rust-analyzer/` directory (if you have a separate rust-analyzer build directory).
- Uninstall unnecessary toolchains if you use `cargo-bisect-rustc`. You can check which toolchains
  are installed with `rustup toolchain list`.

[^1]: issue[#1707](https://github.com/rust-lang/rustc-dev-guide/issues/1707)
