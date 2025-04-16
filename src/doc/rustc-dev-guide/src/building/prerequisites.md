# Prerequisites

## Dependencies

See [the `rust-lang/rust` INSTALL](https://github.com/rust-lang/rust/blob/master/INSTALL.md#dependencies).

## Hardware

You will need an internet connection to build. The bootstrapping process
involves updating git submodules and downloading a beta compiler. It doesn't
need to be super fast, but that can help.

There are no strict hardware requirements, but building the compiler is
computationally expensive, so a beefier machine will help, and I wouldn't
recommend trying to build on a Raspberry Pi! We recommend the following.
* 30GB+ of free disk space. Otherwise, you will have to keep
  clearing incremental caches. More space is better, the compiler is a bit of a
  hog; it's a problem we are aware of.
* 8GB+ RAM
* 2+ cores. Having more cores really helps. 10 or 20 or more is not too many!

Beefier machines will lead to much faster builds. If your machine is not very
powerful, a common strategy is to only use `./x check` on your local machine
and let the CI build test your changes when you push to a PR branch.

Building the compiler takes more than half an hour on my moderately powerful
laptop. We suggest downloading LLVM from CI so you don't have to build it from source
([see here][config]).

Like `cargo`, the build system will use as many cores as possible. Sometimes
this can cause you to run low on memory. You can use `-j` to adjust the number
of concurrent jobs. If a full build takes more than ~45 minutes to an hour, you
are probably spending most of the time swapping memory in and out; try using
`-j1`.

If you don't have too much free disk space, you may want to turn off
incremental compilation ([see here][config]). This will make compilation take
longer (especially after a rebase), but will save a ton of space from the
incremental caches.

[config]: ./how-to-build-and-run.md#create-a-bootstraptoml
