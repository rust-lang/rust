# Getting Started

This documentation is _not_ intended to be comprehensive; it is meant to be a
quick guide for the most useful things. For more information, [see this
chapter](./building/how-to-build-and-run.md).

## Asking Questions

The compiler team (or "t-compiler") usually hangs out in Zulip [in this
"stream"][z]; it will be easiest to get questions answered there.

[z]: https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler

**Please ask questions!** A lot of people report feeling that they are "wasting
expert time", but nobody on t-compiler feels this way. Contributors are
important to us.

Also, if you feel comfortable, prefer public topics, as this means others can
see the questions and answers, and perhaps even integrate them back into this
guide :)

### Experts

Not all `t-compiler` members are experts on all parts of `rustc`; it's a pretty
large project.  To find out who has expertise on different parts of the
compiler, [consult this "experts map"][map].

It's not perfectly complete, though, so please also feel free to ask questions
even if you can't figure out who to ping.

[map]: https://github.com/rust-lang/compiler-team/blob/master/content/experts/map.toml

### Etiquette

We do ask that you be mindful to include as much useful information as you can
in your question, but we recognize this can be hard if you are unfamiliar with
contributing to Rust.

Just pinging someone without providing any context can be a bit annoying and
just create noise, so we ask that you be mindful of the fact that the
`t-compiler` folks get a lot of pings in a day.

## Cloning and Building

The main repository is [`rust-lang/rust`][repo]. This contains the compiler,
the standard library (including `core`, `alloc`, `test`, `proc_macro`, etc),
and a bunch of tools (e.g. `rustdoc`, the bootstrapping infrastructure, etc).

[repo]: https://github.com/rust-lang/rust

There are also a bunch of submodules for things like LLVM, `clippy`, `miri`,
etc. You don't need to clone these immediately, but the build tool will
automatically clone and sync them (more later).

[**Take a look at the "Suggested Workflows" chapter for some helpful
advice.**][suggested]

[suggested]: ./building/suggested.md

### System Requirements

[**See this chapter for detailed software requirements.**](./building/prerequisites.md)
Most notably, you will need Python 2 to run `x.py`.

There are no hard hardware requirements, but building the compiler is
computationally expensive, so a beefier machine will help, and I wouldn't
recommend trying to build on a Raspberry Pi :P

- x86 and ARM are both supported (TODO: confirm)
- Recommended 30GB of free disk space; otherwise, you will have to keep
  clearing incremental caches.
- Recommended >=8GB RAM
- Recommended >=2 cores; more cores really helps
- You will need an internet connection to build; the bootstrapping process
  involves updating git submodules and downloading a beta compiler. It doesn't
  need to be super fast, but that can help.

Building the compiler takes more than half an hour on my moderately powerful
laptop (even longer if you build LLVM).

### Cloning

You can just do a normal git clone:

```shell
git clone https://github.com/rust-lang/rust.git
```

You don't need to clone the submodules at this time.

**Pro tip**: if you contribute often, you may want to look at the git worktrees
tip in [this chapter][suggested].

### Configuring the Compiler

The compiler has a configuration file which contains a ton of settings. We will
provide some recommendations here that should work for most, but [check out
this chapter for more info][config].

[config]: ./building/how-to-build-and-run.html#create-a-configtoml

In the top level of the repo:

```shell
cp config.toml.example config.toml
```

Then, edit `config.toml`. You will need to search for, uncomment, and update
the following settings:

- `debug = true`: enables debug symbols and `debug!` logging, takes a bit longer to compile.
- `incremental = true`: enables incremental compilation of the compiler itself.
  This is turned off by default because it's technically unsound. Sometimes
  this will cause weird crashes, but it can really speed things up.
- `llvm-config`: enable building with system LLVM. [See this chapter][sysllvm]
  for more info. This avoids having to build LLVM, which takes forever.

[sysllvm]: ./building/suggested.html#building-with-system-llvm

### `./x.py` Intro

`rustc` is a bootstrapping compiler because it is written in Rust. Where do you
get the original compiler from? We use the current beta compiler
to build the compiler. Then, we use that compiler to build itself. Thus,
`rustc` has a 2-stage build.

We have a special tool `./x.py` that drives this process. It is used for
compiling the compiler, the standard libraries, and `rustdoc`. It is also used
for driving CI and building the final release artifacts.

### Building and Testing `rustc`

For most contributions, you only need to build stage 1, which saves a lot of time.
After updating `config.toml`, as mentioned above, you can use `./x.py`:

```shell
# Build the compiler (stage 1)
./x.py build --stage 1
```

This will take a while, especially the first time. Be wary of accidentally
touching or formatting the compiler, as `./x.py` will try to recompile it.

To run the compiler's UI test (the bulk of the test suite):

```
# UI tests
./x.py test --stage 1 src/test/ui
```

This will build the compiler first, if needed.

This will be enough for most people. Notably, though, it mostly tests the
compiler frontend, not codegen or debug info.  You can read more about the
different test suites [in this chapter][testing].

[testing]: https://rustc-dev-guide.rust-lang.org/tests/intro.html

If you only want to check that the compiler builds (without actually building
it) you can run the following:

```shell
./x.py check
```

To format the code:

```shell
# Actually format
./x.py fmt

# Just check formatting, exit with error
./x.py fmt --check
```

You can use `RUSTC_LOG=XXX` to get debug logging. [Read more here][logging].

[logging]: ./compiler-debugging.html#getting-logging-output

### Building and Testing `std`/`core`/`alloc`/`test`/`proc_macro`/etc.

TODO

### Building and Testing `rustdoc`

TODO

### Contributing code to other Rust projects

TODO: talk about things like miri, clippy, chalk, etc

## Contributor Procedures

There are some official procedures to know about. This is a tour of the
highlights, but there are a lot more details, which we will link to below.

### Bug Fixes

TODO: talk about bors, highfive

### New Features

TODO: talk about RFCs, stabilization

### Breaking Changes

TODO: talk about crater, FCP, etc

### Major Changes

TODO: talk about MCP

### Performance

TODO: Talk about perf runs

## Other Resources

- This guide: talks about how `rustc` works
- [The t-compiler zulip][z]
- [The compiler's documentation (rustdocs)](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/)

TODO: am I missing any?
