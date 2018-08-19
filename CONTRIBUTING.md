# Contributing to Rust
[contributing-to-rust]: #contributing-to-rust

Thank you for your interest in contributing to Rust! There are many ways to
contribute, and we appreciate all of them. This document is a bit long, so here's
links to the major sections:

* [Feature Requests](#feature-requests)
* [Bug Reports](#bug-reports)
* [The Build System](#the-build-system)
* [Pull Requests](#pull-requests)
* [Writing Documentation](#writing-documentation)
* [Issue Triage](#issue-triage)
* [Out-of-tree Contributions](#out-of-tree-contributions)
* [Helpful Links and Information](#helpful-links-and-information)

If you have questions, please make a post on [internals.rust-lang.org][internals] or
hop on [#rust-internals][pound-rust-internals].

As a reminder, all contributors are expected to follow our [Code of Conduct][coc].

[pound-rust-internals]: https://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust-internals
[internals]: https://internals.rust-lang.org
[coc]: https://www.rust-lang.org/conduct.html

## Feature Requests
[feature-requests]: #feature-requests

To request a change to the way the Rust language works, please head over
to the [RFCs repository](https://github.com/rust-lang/rfcs) and view the 
[README](https://github.com/rust-lang/rfcs/blob/master/README.md)
for instructions.

## Bug Reports
[bug-reports]: #bug-reports

While bugs are unfortunate, they're a reality in software. We can't fix what we
don't know about, so please report liberally. If you're not sure if something
is a bug or not, feel free to file a bug anyway.

**If you believe reporting your bug publicly represents a security risk to Rust users,
please follow our [instructions for reporting security vulnerabilities](https://www.rust-lang.org/security.html)**.

If you have the chance, before reporting a bug, please [search existing
issues](https://github.com/rust-lang/rust/search?q=&type=Issues&utf8=%E2%9C%93),
as it's possible that someone else has already reported your error. This doesn't
always work, and sometimes it's hard to know what to search for, so consider this
extra credit. We won't mind if you accidentally file a duplicate report.

Similarly, to help others who encountered the bug find your issue,
consider filing an issue with a descriptive title, which contains information that might be unique to it.
This can be the language or compiler feature used, the conditions that trigger the bug,
or part of the error message if there is any.
An example could be: **"impossible case reached" on lifetime inference for impl Trait in return position**.

Opening an issue is as easy as following [this
link](https://github.com/rust-lang/rust/issues/new) and filling out the fields.
Here's a template that you can use to file a bug, though it's not necessary to
use it exactly:

    <short summary of the bug>

    I tried this code:

    <code sample that causes the bug>

    I expected to see this happen: <explanation>

    Instead, this happened: <explanation>

    ## Meta

    `rustc --version --verbose`:

    Backtrace:

All three components are important: what you did, what you expected, what
happened instead. Please include the output of `rustc --version --verbose`,
which includes important information about what platform you're on, what
version of Rust you're using, etc.

Sometimes, a backtrace is helpful, and so including that is nice. To get
a backtrace, set the `RUST_BACKTRACE` environment variable to a value
other than `0`. The easiest way
to do this is to invoke `rustc` like this:

```bash
$ RUST_BACKTRACE=1 rustc ...
```

## The Build System
[the-build-system]: #the-build-system

Rust's build system allows you to bootstrap the compiler, run tests &
benchmarks, generate documentation, install a fresh build of Rust, and more.
It's your best friend when working on Rust, allowing you to compile & test
your contributions before submission.

The build system lives in [the `src/bootstrap` directory][bootstrap] in the
project root. Our build system is itself written in Rust and is based on Cargo
to actually build all the compiler's crates. If you have questions on the build
system internals, try asking in [`#rust-internals`][pound-rust-internals].

[bootstrap]: https://github.com/rust-lang/rust/tree/master/src/bootstrap/

### Configuration
[configuration]: #configuration

Before you can start building the compiler you need to configure the build for
your system. In most cases, that will just mean using the defaults provided
for Rust.

To change configuration, you must copy the file `config.toml.example`
to `config.toml` in the directory from which you will be running the build, and
change the settings provided.

There are large number of options provided in this config file that will alter the
configuration used in the build process. Some options to note:

#### `[llvm]`:
- `assertions = true` = This enables LLVM assertions, which makes LLVM misuse cause an assertion failure instead of weird misbehavior. This also slows down the compiler's runtime by ~20%.
- `ccache = true` - Use ccache when building llvm

#### `[build]`:
- `compiler-docs = true` - Build compiler documentation

#### `[rust]`:
- `debuginfo = true` - Build a compiler with debuginfo. Makes building rustc slower, but then you can use a debugger to debug `rustc`.
- `debuginfo-lines = true` - An alternative to `debuginfo = true` that doesn't let you use a debugger, but doesn't make building rustc slower and still gives you line numbers in backtraces.
- `debuginfo-tools = true` - Build the extended tools with debuginfo.
- `debug-assertions = true` - Makes the log output of `debug!` work.
- `optimize = false` - Disable optimizations to speed up compilation of stage1 rust, but makes the stage1 compiler x100 slower.

For more options, the `config.toml` file contains commented out defaults, with
descriptions of what each option will do.

Note: Previously the `./configure` script was used to configure this
project. It can still be used, but it's recommended to use a `config.toml`
file. If you still have a `config.mk` file in your directory - from
`./configure` - you may need to delete it for `config.toml` to work.

### Building
[building]: #building

A default configuration requires around 3.5 GB of disk space, whereas building a debug configuration may require more than 30 GB.

Dependencies
- [build dependencies](README.md#building-from-source)
- `gdb` 6.2.0 minimum, 7.1 or later recommended for test builds

The build system uses the `x.py` script to control the build process. This script
is used to build, test, and document various parts of the compiler. You can
execute it as:

```sh
python x.py build
```

On some systems you can also use the shorter version:

```sh
./x.py build
```

To learn more about the driver and top-level targets, you can execute:

```sh
python x.py --help
```

The general format for the driver script is:

```sh
python x.py <command> [<directory>]
```

Some example commands are `build`, `test`, and `doc`. These will build, test,
and document the specified directory. The second argument, `<directory>`, is
optional and defaults to working over the entire compiler. If specified,
however, only that specific directory will be built. For example:

```sh
# build the entire compiler
python x.py build

# build all documentation
python x.py doc

# run all test suites
python x.py test

# build only the standard library
python x.py build src/libstd

# test only one particular test suite
python x.py test src/test/rustdoc

# build only the stage0 libcore library
python x.py build src/libcore --stage 0
```

You can explore the build system through the various `--help` pages for each
subcommand. For example to learn more about a command you can run:

```
python x.py build --help
```

To learn about all possible rules you can execute, run:

```
python x.py build --help --verbose
```

Note: Previously `./configure` and `make` were used to build this project.
They are still available, but `x.py` is the recommended build system.

### Useful commands
[useful-commands]: #useful-commands

Some common invocations of `x.py` are:

- `x.py build --help` - show the help message and explain the subcommand
- `x.py build src/libtest --stage 1` - build up to (and including) the first
  stage. For most cases we don't need to build the stage2 compiler, so we can
  save time by not building it. The stage1 compiler is a fully functioning
  compiler and (probably) will be enough to determine if your change works as
  expected.
- `x.py build src/rustc --stage 1` - This will build just rustc, without libstd.
  This is the fastest way to recompile after you changed only rustc source code.
  Note however that the resulting rustc binary won't have a stdlib to link
  against by default. You can build libstd once with `x.py build src/libstd`,
  but it is only guaranteed to work if recompiled, so if there are any issues
  recompile it.
- `x.py test` - build the full compiler & run all tests (takes a while). This
  is what gets run by the continuous integration system against your pull
  request. You should run this before submitting to make sure your tests pass
  & everything builds in the correct manner.
- `x.py test src/libstd --stage 1` - test the standard library without
  recompiling stage 2.
- `x.py test src/test/run-pass --test-args TESTNAME` - Run a matching set of
  tests.
  - `TESTNAME` should be a substring of the tests to match against e.g. it could
    be the fully qualified test name, or just a part of it.
    `TESTNAME=collections::hash::map::test_map::test_capacity_not_less_than_len`
    or `TESTNAME=test_capacity_not_less_than_len`.
- `x.py test src/test/run-pass --stage 1 --test-args <substring-of-test-name>` -
  Run a single rpass test with the stage1 compiler (this will be quicker than
  running the command above as we only build the stage1 compiler, not the entire
  thing).  You can also leave off the directory argument to run all stage1 test
  types.
- `x.py test src/libcore --stage 1` - Run stage1 tests in `libcore`.
- `x.py test src/tools/tidy` - Check that the source code is in compliance with
  Rust's style guidelines. There is no official document describing Rust's full
  guidelines as of yet, but basic rules like 4 spaces for indentation and no
  more than 99 characters in a single line should be kept in mind when writing
  code.

### Using your local build
[using-local-build]: #using-local-build

If you use Rustup to manage your rust install, it has a feature called ["custom
toolchains"][toolchain-link] that you can use to access your newly-built compiler
without having to install it to your system or user PATH. If you've run `python
x.py build`, then you can add your custom rustc to a new toolchain like this:

[toolchain-link]: https://github.com/rust-lang-nursery/rustup.rs#working-with-custom-toolchains-and-local-builds

```
rustup toolchain link <name> build/<host-triple>/stage2
```

Where `<host-triple>` is the build triple for the host (the triple of your
computer, by default), and `<name>` is the name for your custom toolchain. (If you
added `--stage 1` to your build command, the compiler will be in the `stage1`
folder instead.) You'll only need to do this once - it will automatically point
to the latest build you've done.

Once this is set up, you can use your custom toolchain just like any other. For
example, if you've named your toolchain `local`, running `cargo +local build` will
compile a project with your custom rustc, setting `rustup override set local` will
override the toolchain for your current directory, and `cargo +local doc` will use
your custom rustc and rustdoc to generate docs. (If you do this with a `--stage 1`
build, you'll need to build rustdoc specially, since it's not normally built in
stage 1. `python x.py build --stage 1 src/libstd src/tools/rustdoc` will build
rustdoc and libstd, which will allow rustdoc to be run with that toolchain.)

### Out-of-tree builds
[out-of-tree-builds]: #out-of-tree-builds

Rust's `x.py` script fully supports out-of-tree builds - it looks for
the Rust source code from the directory `x.py` was found in, but it
reads the `config.toml` configuration file from the directory it's
run in, and places all build artifacts within a subdirectory named `build`.

This means that if you want to do an out-of-tree build, you can just do it:
```
$ cd my/build/dir
$ cp ~/my-config.toml config.toml # Or fill in config.toml otherwise
$ path/to/rust/x.py build
...
$ # This will use the Rust source code in `path/to/rust`, but build
$ # artifacts will now be in ./build
```

It's absolutely fine to have multiple build directories with different
`config.toml` configurations using the same code.

## Pull Requests
[pull-requests]: #pull-requests

Pull requests are the primary mechanism we use to change Rust. GitHub itself
has some [great documentation][about-pull-requests] on using the Pull Request feature.
We use the "fork and pull" model [described here][development-models], where
contributors push changes to their personal fork and create pull requests to
bring those changes into the source repository.

[about-pull-requests]: https://help.github.com/articles/about-pull-requests/
[development-models]: https://help.github.com/articles/about-collaborative-development-models/

Please make pull requests against the `master` branch.

Compiling all of `./x.py test` can take a while. When testing your pull request,
consider using one of the more specialized `./x.py` targets to cut down on the
amount of time you have to wait. You need to have built the compiler at least
once before running these will work, but that’s only one full build rather than
one each time.

    $ python x.py test --stage 1

is one such example, which builds just `rustc`, and then runs the tests. If
you’re adding something to the standard library, try

    $ python x.py test src/libstd --stage 1

Please make sure your pull request is in compliance with Rust's style
guidelines by running

    $ python x.py test src/tools/tidy

Make this check before every pull request (and every new commit in a pull
request) ; you can add [git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
before every push to make sure you never forget to make this check.

All pull requests are reviewed by another person. We have a bot,
@rust-highfive, that will automatically assign a random person to review your
request.

If you want to request that a specific person reviews your pull request,
you can add an `r?` to the message. For example, Steve usually reviews
documentation changes. So if you were to make a documentation change, add

    r? @steveklabnik

to the end of the message, and @rust-highfive will assign @steveklabnik instead
of a random person. This is entirely optional.

After someone has reviewed your pull request, they will leave an annotation
on the pull request with an `r+`. It will look something like this:

    @bors: r+ 38fe8d2

This tells @bors, our lovable integration bot, that your pull request has
been approved. The PR then enters the [merge queue][merge-queue], where @bors
will run all the tests on every platform we support. If it all works out,
@bors will merge your code into `master` and close the pull request.

[merge-queue]: https://buildbot2.rust-lang.org/homu/queue/rust

Speaking of tests, Rust has a comprehensive test suite. More information about
it can be found
[here](https://github.com/rust-lang/rust/blob/master/src/test/COMPILER_TESTS.md).

### External Dependencies
[external-dependencies]: #external-dependencies

Currently building Rust will also build the following external projects:

* [clippy](https://github.com/rust-lang-nursery/rust-clippy)
* [miri](https://github.com/solson/miri)
* [rustfmt](https://github.com/rust-lang-nursery/rustfmt)
* [rls](https://github.com/rust-lang-nursery/rls/)

We allow breakage of these tools in the nightly channel. Maintainers of these
projects will be notified of the breakages and should fix them as soon as
possible.

After the external is fixed, one could add the changes with

```sh
git add path/to/submodule
```

outside the submodule.

In order to prepare your tool-fixing PR, you can run the build locally by doing
`./x.py build src/tools/TOOL`. If you will be editing the sources
there, you may wish to set `submodules = false` in the `config.toml`
to prevent `x.py` from resetting to the original branch.

Breakage is not allowed in the beta and stable channels, and must be addressed
before the PR is merged.

#### Breaking Tools Built With The Compiler
[breaking-tools-built-with-the-compiler]: #breaking-tools-built-with-the-compiler

Rust's build system builds a number of tools that make use of the
internals of the compiler. This includes clippy,
[RLS](https://github.com/rust-lang-nursery/rls) and
[rustfmt](https://github.com/rust-lang-nursery/rustfmt). If these tools
break because of your changes, you may run into a sort of "chicken and egg"
problem. These tools rely on the latest compiler to be built so you can't update
them to reflect your changes to the compiler until those changes are merged into
the compiler. At the same time, you can't get your changes merged into the compiler
because the rust-lang/rust build won't pass until those tools build and pass their
tests.

That means that, in the default state, you can't update the compiler without first
fixing rustfmt, rls and the other tools that the compiler builds.

Luckily, a feature was [added to Rust's build](https://github.com/rust-lang/rust/issues/45861)
to make all of this easy to handle. The idea is that we allow these tools to be "broken",
so that the rust-lang/rust build passes without trying to build them, then land the change
in the compiler, wait for a nightly, and go update the tools that you broke. Once you're done
and the tools are working again, you go back in the compiler and update the tools
so they can be distributed again.

This should avoid a bunch of synchronization dances and is also much easier on contributors as
there's no need to block on rls/rustfmt/other tools changes going upstream.

Here are those same steps in detail:

1. (optional) First, if it doesn't exist already, create a `config.toml` by copying
   `config.toml.example` in the root directory of the Rust repository.
   Set `submodules = false` in the `[build]` section. This will prevent `x.py`
   from resetting to the original branch after you make your changes. If you
   need to [update any submodules to their latest versions][updating-submodules],
   see the section of this file about that for more information.
2. (optional) Run `./x.py test src/tools/rustfmt` (substituting the submodule
   that broke for `rustfmt`). Fix any errors in the submodule (and possibly others).
3. (optional) Make commits for your changes and send them to upstream repositories as a PR.
4. (optional) Maintainers of these submodules will **not** merge the PR. The PR can't be
   merged because CI will be broken. You'll want to write a message on the PR referencing
   your change, and how the PR should be merged once your change makes it into a nightly.
5. Wait for your PR to merge.
6. Wait for a nightly
7. (optional) Help land your PR on the upstream repository now that your changes are in nightly.
8. (optional) Send a PR to rust-lang/rust updating the submodule.

#### Updating submodules
[updating-submodules]: #updating-submodules

These instructions are specific to updating `rustfmt`, however they may apply
to the other submodules as well. Please help by improving these instructions
if you find any discrepancies or special cases that need to be addressed.

To update the `rustfmt` submodule, start by running the appropriate
[`git submodule` command](https://git-scm.com/book/en/v2/Git-Tools-Submodules).
For example, to update to the latest commit on the remote master branch,
you may want to run:
```
git submodule update --remote src/tools/rustfmt
```
If you run `./x.py build` now, and you are lucky, it may just work. If you see
an error message about patches that did not resolve to any crates, you will need
to complete a few more steps which are outlined with their rationale below.

*(This error may change in the future to include more information.)*
```
error: failed to resolve patches for `https://github.com/rust-lang-nursery/rustfmt`

Caused by:
  patch for `rustfmt-nightly` in `https://github.com/rust-lang-nursery/rustfmt` did not resolve to any crates
failed to run: ~/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo build --manifest-path ~/rust/src/bootstrap/Cargo.toml
```

If you haven't used the `[patch]`
section of `Cargo.toml` before, there is [some relevant documentation about it
in the cargo docs](http://doc.crates.io/manifest.html#the-patch-section). In
addition to that, you should read the
[Overriding dependencies](http://doc.crates.io/specifying-dependencies.html#overriding-dependencies)
section of the documentation as well.

Specifically, the following [section in Overriding dependencies](http://doc.crates.io/specifying-dependencies.html#testing-a-bugfix) reveals what the problem is:

> Next up we need to ensure that our lock file is updated to use this new version of uuid so our project uses the locally checked out copy instead of one from crates.io. The way [patch] works is that it'll load the dependency at ../path/to/uuid and then whenever crates.io is queried for versions of uuid it'll also return the local version.
>
> This means that the version number of the local checkout is significant and will affect whether the patch is used. Our manifest declared uuid = "1.0" which means we'll only resolve to >= 1.0.0, < 2.0.0, and Cargo's greedy resolution algorithm also means that we'll resolve to the maximum version within that range. Typically this doesn't matter as the version of the git repository will already be greater or match the maximum version published on crates.io, but it's important to keep this in mind!

This says that when we updated the submodule, the version number in our
`src/tools/rustfmt/Cargo.toml` changed. The new version is different from
the version in `Cargo.lock`, so the build can no longer continue.

To resolve this, we need to update `Cargo.lock`. Luckily, cargo provides a
command to do this easily.

First, go into the `src/` directory since that is where `Cargo.toml` is in
the rust repository. Then run, `cargo update -p rustfmt-nightly` to solve
the problem.

```
$ cd src
$ cargo update -p rustfmt-nightly
```

This should change the version listed in `src/Cargo.lock` to the new version you updated
the submodule to. Running `./x.py build` should work now.

## Writing Documentation
[writing-documentation]: #writing-documentation

Documentation improvements are very welcome. The source of `doc.rust-lang.org`
is located in `src/doc` in the tree, and standard API documentation is generated
from the source code itself.

Documentation pull requests function in the same way as other pull requests,
though you may see a slightly different form of `r+`:

    @bors: r+ 38fe8d2 rollup

That additional `rollup` tells @bors that this change is eligible for a 'rollup'.
To save @bors some work, and to get small changes through more quickly, when
@bors attempts to merge a commit that's rollup-eligible, it will also merge
the other rollup-eligible patches too, and they'll get tested and merged at
the same time.

To find documentation-related issues, sort by the [T-doc label][tdoc].

[tdoc]: https://github.com/rust-lang/rust/issues?q=is%3Aopen%20is%3Aissue%20label%3AT-doc

You can find documentation style guidelines in [RFC 1574][rfc1574].

[rfc1574]: https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md#appendix-a-full-conventions-text

In many cases, you don't need a full `./x.py doc`. You can use `rustdoc` directly
to check small fixes. For example, `rustdoc src/doc/reference.md` will render
reference to `doc/reference.html`. The CSS might be messed up, but you can
verify that the HTML is right.

## Issue Triage
[issue-triage]: #issue-triage

Sometimes, an issue will stay open, even though the bug has been fixed. And
sometimes, the original bug may go stale because something has changed in the
meantime.

It can be helpful to go through older bug reports and make sure that they are
still valid. Load up an older issue, double check that it's still true, and
leave a comment letting us know if it is or is not. The [least recently
updated sort][lru] is good for finding issues like this.

Contributors with sufficient permissions on the Rust repo can help by adding
labels to triage issues:

* Yellow, **A**-prefixed labels state which **area** of the project an issue
  relates to.

* Magenta, **B**-prefixed labels identify bugs which are **blockers**.

* Dark blue, **beta-** labels track changes which need to be backported into
  the beta branches.

* Light purple, **C**-prefixed labels represent the **category** of an issue.

* Green, **E**-prefixed labels explain the level of **experience** necessary
  to fix the issue.

* The dark blue **final-comment-period** label marks bugs that are using the
  RFC signoff functionality of [rfcbot][rfcbot] and are currenty in the final
  comment period.

* Red, **I**-prefixed labels indicate the **importance** of the issue. The
  [I-nominated][inom] label indicates that an issue has been nominated for
  prioritizing at the next triage meeting.

* The purple **metabug** label marks lists of bugs collected by other
  categories.

* Purple gray, **O**-prefixed labels are the **operating system** or platform
  that this issue is specific to.

* Orange, **P**-prefixed labels indicate a bug's **priority**. These labels
  are only assigned during triage meetings, and replace the [I-nominated][inom]
  label.

* The gray **proposed-final-comment-period** label marks bugs that are using
  the RFC signoff functionality of [rfcbot][rfcbot] and are currently awaiting
  signoff of all team members in order to enter the final comment period.

* Pink, **regression**-prefixed labels track regressions from stable to the
  release channels.

* The light orange **relnotes** label marks issues that should be documented in
  the release notes of the next release.

* Gray, **S**-prefixed labels are used for tracking the **status** of pull
  requests.

* Blue, **T**-prefixed bugs denote which **team** the issue belongs to.

If you're looking for somewhere to start, check out the [E-easy][eeasy] tag.

[inom]: https://github.com/rust-lang/rust/issues?q=is%3Aopen+is%3Aissue+label%3AI-nominated
[eeasy]: https://github.com/rust-lang/rust/issues?q=is%3Aopen+is%3Aissue+label%3AE-easy
[lru]: https://github.com/rust-lang/rust/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-asc
[rfcbot]: https://github.com/anp/rfcbot-rs/

## Out-of-tree Contributions
[out-of-tree-contributions]: #out-of-tree-contributions

There are a number of other ways to contribute to Rust that don't deal with
this repository.

Answer questions in [#rust][pound-rust], or on [users.rust-lang.org][users],
or on [StackOverflow][so].

Participate in the [RFC process](https://github.com/rust-lang/rfcs).

Find a [requested community library][community-library], build it, and publish
it to [Crates.io](http://crates.io). Easier said than done, but very, very
valuable!

[pound-rust]: http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust
[users]: https://users.rust-lang.org/
[so]: http://stackoverflow.com/questions/tagged/rust
[community-library]: https://github.com/rust-lang/rfcs/labels/A-community-library

## Helpful Links and Information
[helpful-info]: #helpful-info

For people new to Rust, and just starting to contribute, or even for
more seasoned developers, some useful places to look for information
are:

* The [rustc guide] contains information about how various parts of the compiler work
* [Rust Forge][rustforge] contains additional documentation, including write-ups of how to achieve common tasks
* The [Rust Internals forum][rif], a place to ask questions and
  discuss Rust's internals
* The [generated documentation for rust's compiler][gdfrustc]
* The [rust reference][rr], even though it doesn't specifically talk about Rust's internals, it's a great resource nonetheless
* Although out of date, [Tom Lee's great blog article][tlgba] is very helpful
* [rustaceans.org][ro] is helpful, but mostly dedicated to IRC
* The [Rust Compiler Testing Docs][rctd]
* For @bors, [this cheat sheet][cheatsheet] is helpful (Remember to replace `@homu` with `@bors` in the commands that you use.)
* **Google!** ([search only in Rust Documentation][gsearchdocs] to find types, traits, etc. quickly)
* Don't be afraid to ask! The Rust community is friendly and helpful.

[rustc guide]: https://rust-lang-nursery.github.io/rustc-guide/about-this-guide.html
[gdfrustc]: http://manishearth.github.io/rust-internals-docs/rustc/
[gsearchdocs]: https://www.google.com/search?q=site:doc.rust-lang.org+your+query+here
[rif]: http://internals.rust-lang.org
[rr]: https://doc.rust-lang.org/book/README.html
[rustforge]: https://forge.rust-lang.org/
[tlgba]: http://tomlee.co/2014/04/a-more-detailed-tour-of-the-rust-compiler/
[ro]: http://www.rustaceans.org/
[rctd]: ./src/test/COMPILER_TESTS.md
[cheatsheet]: https://buildbot2.rust-lang.org/homu/
