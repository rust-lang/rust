# Getting Started

<!-- toc -->

This documentation is _not_ intended to be comprehensive; it is meant to be a
quick guide for the most useful things. For more information, [see this
chapter on how to build and run the compiler](./building/how-to-build-and-run.md).

## Asking Questions

The compiler team (or `t-compiler`) usually hangs out in Zulip [in this
"stream"][z]; it will be easiest to get questions answered there.

[z]: https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler

**Please ask questions!** A lot of people report feeling that they are "wasting
expert time", but nobody on `t-compiler` feels this way. Contributors are
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
automatically clone and sync them (more on this later).

[**Take a look at the "Suggested Workflows" chapter for some helpful
advice.**][suggested]

[suggested]: ./building/suggested.md

### System Requirements

[**See this chapter for detailed software requirements.**](./building/prerequisites.md)
Most notably, you will need Python 2 or 3 to run `x.py`.

There are no hard hardware requirements, but building the compiler is
computationally expensive, so a beefier machine will help, and I wouldn't
recommend trying to build on a Raspberry Pi :P

- Recommended >=30GB of free disk space; otherwise, you will have to keep
  clearing incremental caches. More space is better, the compiler is a bit of a
  hog; it's a problem we are aware of.
- Recommended >=8GB RAM.
- Recommended >=2 cores; having more cores really helps.
- You will need an internet connection to build; the bootstrapping process
  involves updating git submodules and downloading a beta compiler. It doesn't
  need to be super fast, but that can help.

Building the compiler takes more than half an hour on my moderately powerful
laptop. The first time you build the compiler, LLVM will also be built unless
you use your system's LLVM ([see below][configsec]).

[configsec]: #configuring-the-compiler

Like `cargo`, the build system will use as many cores as possible. Sometimes
this can cause you to run low on memory. You can use `-j` to adjust the number
concurrent jobs. If a full build takes more than ~45 minutes to an hour,
you are probably spending most of the time swapping memory in and out;
try using `-j1`.

On a slow machine, the build times for rustc are very painful. Consider using
`./x.py check` instead of a full build and letting the automated tests run
when you push to GitHub.

If you don't have too much free disk space, you may want to turn off
incremental compilation ([see below][configsec]). This will make
compilation take longer (especially after a rebase),
but will save a ton of space from the incremental caches.

### Cloning

You can just do a normal git clone:

```sh
git clone https://github.com/rust-lang/rust.git
```

You don't need to clone the submodules at this time. But if you want to, you
can do the following:

```sh
# first time
git submodule update --init --recursive

# subsequent times (to pull new commits)
git submodule update
```

### Configuring the Compiler

The compiler has a configuration file which contains a ton of settings. We will
provide some recommendations here that should work for most, but [check out
this chapter for more info][config].

[config]: ./building/how-to-build-and-run.html#create-a-configtoml

In the top level of the repo:

```sh
$ x.py setup
```

This will walk you through an interactive setup for x.py that looks like this:

```
$ x.py setup
Welcome to the Rust project! What do you want to do with x.py?
a) Contribute to the standard library
b) Contribute to the compiler
c) Contribute to the compiler, and also modify LLVM or codegen
d) Install Rust from source
Please choose one (a/b/c/d): a
`x.py` will now use the configuration at /home/joshua/rustc2/src/bootstrap/defaults/config.toml.library
To get started, try one of the following commands:
- `x.py check`
- `x.py build`
- `x.py test library/std`
- `x.py doc`
For more suggestions, see https://rustc-dev-guide.rust-lang.org/building/suggested.html
```

You may also want to set up [system LLVM][sysllvm] to avoid building LLVM from source.

[sysllvm]: ./building/suggested.html#skipping-llvm-build

### `./x.py` Intro

`rustc` is a _bootstrapping_ compiler, which means that it is written in Rust
and thus needs to be compiled by itself. So where do you
get the original compiler from? We use the current beta compiler
to build a new compiler. Then, we use that compiler to build itself. Thus,
`rustc` has a 2-stage build. You can read more about bootstrapping
[here][boot], but you don't need to know much more to contribute.

[boot]: ./building/bootstrapping.md

We have a special tool `./x.py` that drives this process. It is used for
building the compiler, the standard libraries, and `rustdoc`. It is also used
for driving CI and building the final release artifacts.

Unfortunately, a proper 2-stage build takes a long time depending on your
hardware, but it is the only correct way to build everything (e.g. it's what
the CI and release processes use). **However, in most cases, you can get by
without a full 2-stage build**. In the following section, we give instructions
for how to do "the correct thing", but then we also give various tips to speed
things up.

### Building and Testing `rustc`

Here is a summary of the different commands for reference, but you probably
should still read the rest of the section:

| Command | When to use it |
| --- | --- |
| `x.py check` | Quick check to see if things compile; [rust-analyzer can run this automatically for you][rust-analyzer] |
| `x.py build --stage 0 [library/std]` | Build only the standard library, without building the compiler |
| `x.py build library/std` | Build just the 1st stage of the compiler, along with the standard library; this is faster than building stage 2 and usually good enough |
| `x.py build --keep-stage 1 library/std` | Build the 1st stage of the compiler and skips rebuilding the standard library; this is useful after you've done an ordinary stage1 build to skip compilation time, but it can cause weird problems. (Just do a regular build to resolve.) |
| `x.py test [--keep-stage 1]` | Run the test suite using the stage1 compiler |
| `x.py test --bless [--keep-stage 1]` | Run the test suite using the stage1 compiler _and_ update expected test output. |
| `x.py build --stage 2 compiler/rustc` | Do a full 2-stage build. You almost never want to do this. |
| `x.py test --stage 2` | Do a full 2-stage build and run all tests. You almost never want to do this. |

To do a full 2-stage build of the whole compiler, you should run this (after
updating `config.toml` as mentioned above):

```sh
./x.py build --stage 2 compiler/rustc
```

In the process, this will also necessarily build the standard libraries, and it
will build `rustdoc` (which doesn't take too long).

To build and test everything:

```sh
./x.py test
```

For most contributions, you only need to build stage 1, which saves a lot of time:

```sh
# Build the compiler (stage 1)
./x.py build library/std

# Subsequent builds
./x.py build --keep-stage 1 library/std
```

This will take a while, especially the first time. Be wary of accidentally
touching or formatting the compiler, as `./x.py` will try to recompile it.

**NOTE**: The `--keep-stage 1` will _assume_ that the stage 0 standard library
does not need to be rebuilt, which is usually true, which will save some time.
However, if you are changing certain parts of the compiler, this may lead to
weird errors. Feel free to ask on [zulip][z] if you are running into issues.

This runs a ton of tests and takes a long time to complete. If you are
working on `rustc`, you can usually get by with only the [UI tests][uitests]. These
test are mostly for the frontend of the compiler, so if you are working on LLVM
or codegen, this shortcut will _not_ test your changes. You can read more about the
different test suites [in this chapter][testing].

[rust-analyzer]: ./building/suggested.html#configuring-rust-analyzer-for-rustc
[uitests]: ./tests/adding.html#ui
[testing]: https://rustc-dev-guide.rust-lang.org/tests/intro.html

```sh
# First build
./x.py test src/test/ui

# Subsequent builds
./x.py test src/test/ui --keep-stage 1
```

If your changes impact test output, you can use `--bless` to automatically
update the `.stderr` files of the affected tests:

```sh
./x.py test src/test/ui --keep-stage 1 --bless
```

While working on the compiler, it can be helpful to see if the code just
compiles (similar to `cargo check`) without actually building it. You can do
this with:

```sh
./x.py check
```

This command is really fast (relative to the other commands). It usually
completes in a couple of minutes on my laptop. **A common workflow when working
on the compiler is to make changes and repeatedly check with `./x.py check`.
Then, run the tests as shown above when you think things should work.**

Finally, the CI ensures that the codebase is using consistent style. To format
the code:

```sh
# Actually format
./x.py fmt

# Just check formatting, exit with error
./x.py fmt --check
```

*Note*: we don't use stable `rustfmt`; we use a pinned version with a special
config, so this may result in different style from normal `rustfmt` if you have
format-on-save turned on. It's a good habit to run `./x.py fmt` before every
commit, as this reduces conflicts later. The pinned version is built under
`build/<target>/stage0/bin/rustfmt`, so if you want, you can use it for a
single file or for format-on-save in your editor, which can be faster than `./x.py fmt`.

One last thing: you can use `RUSTC_LOG=XXX` to get debug logging. [Read more
here][logging]. Notice the `C` in `RUSTC_LOG`. Other than that, it uses normal
[`env_logger`][envlog] syntax.

[envlog]: https://crates.io/crates/env_logger
[logging]: ./compiler-debugging.html#getting-logging-output

### Building and Testing `std`/`core`/`alloc`/`test`/`proc_macro`/etc.

As before, technically the proper way to build one of these libraries is to use
the stage-2 compiler, which of course requires a 2-stage build, described above
(`./x.py build`).

In practice, though, you don't need to build the compiler unless you are
planning to use a recently added nightly feature. Instead, you can just build
stage 0, which uses the current beta compiler.

```sh
./x.py build --stage 0
```

```sh
./x.py test --stage 0 library/std
```

(The same works for `library/alloc`, `library/core`, etc.)

### Building and Testing `rustdoc`

`rustdoc` uses `rustc` internals (and, of course, the standard library), so you
will have to build the compiler and `std` once before you can build `rustdoc`.
As before, you can use `./x.py build` to do this. The first time you build,
the stage-1 compiler will also be built.

```sh
# First build
./x.py build

# Subsequent builds
./x.py build --keep-stage 1
```

As with the compiler, you can do a fast check build:

```sh
./x.py check
```

Rustdoc has two types of tests: content tests and UI tests.

```sh
# Content tests
./x.py test src/test/rustdoc

# UI tests
./x.py test src/test/rustdoc-ui

# Both at once
./x.py test src/test/rustdoc src/test/rustdoc-ui
```

### Contributing code to other Rust projects

There are a bunch of other projects that you can contribute to outside of the
`rust-lang/rust` repo, including `clippy`, `miri`, `chalk`, and many others.

These repos might have their own contributing guidelines and procedures. Many
of them are owned by working groups (e.g. `chalk` is largely owned by
WG-traits). For more info, see the documentation in those repos' READMEs.

### Other ways to contribute

There are a bunch of other ways you can contribute, especially if you don't
feel comfortable jumping straight into the large `rust-lang/rust` codebase.

The following tasks are doable without much background knowledge but are
incredibly helpful:

- [Cleanup crew][iceb]: find minimal reproductions of ICEs, bisect
  regressions, etc. This is a way of helping that saves a ton of time for
  others to fix an error later.
- [Writing documentation][wd]: if you are feeling a bit more intrepid, you could try
  to read a part of the code and write doc comments for it. This will help you
  to learn some part of the compiler while also producing a useful artifact!
- [Working groups][wg]: there are a bunch of working groups on a wide variety
  of rust-related things.

[iceb]: ./notification-groups/cleanup-crew.md
[wd]: ./contributing.md#writing-documentation
[wg]: https://rust-lang.github.io/compiler-team/working-groups/


## Contributor Procedures

There are some official procedures to know about. This is a tour of the
highlights, but there are a lot more details, which we will link to below.

### Code Review

When you open a PR on the `rust-lang/rust` repo, a bot called `@rust-highfive` will
automatically assign a reviewer to the PR. The reviewer is the person that will
approve the PR to be tested and merged. If you want a specific reviewer (e.g. a
team member you've been working with), you can specifically request them by
writing `r? @user` (e.g. `r? @eddyb`) in either the original post or a followup
comment (you can see [this comment][r?] for example).

Please note that the reviewers are humans, who for the most part work on `rustc`
in their free time. This means that they can take some time to respond and review
your PR. It also means that reviewers can miss some PRs that are assigned to them.

To try to move PRs forward, the Triage WG regularly goes through all PRs that
are waiting for review and haven't been discussed for at least 2 weeks. If you
don't get a review within 2 weeks, feel free to ask the Triage WG on
Zulip ([#t-release/triage]). They have knowledge of when to ping, who might be
on vacation, etc.

The reviewer may request some changes using the GitHub code review interface.
They may also request special procedures (such as a [crater] run; [see
below][break]) for some PRs.

[r?]: https://github.com/rust-lang/rust/pull/78133#issuecomment-712692371
[#t-release/triage]: https://rust-lang.zulipchat.com/#narrow/stream/242269-t-release.2Ftriage
[break]: #breaking-changes

When the PR is ready to be merged, the reviewer will issue a command to
`@bors`, the CI bot. Usually, this is `@bors r+` or `@bors r=user` to approve
a PR (there are few other commands, but they are less relevant here).
You can see [this comment][r+] for example. This puts the PR in [bors's queue][bors]
to be tested and merged. Be patient; this can take a while and the queue can
sometimes be long. PRs are never merged by hand.

[r+]: https://github.com/rust-lang/rust/pull/78133#issuecomment-712726339
[bors]: https://bors.rust-lang.org/queue/rust

### Bug Fixes or "Normal" code changes

For most PRs, no special procedures are needed. You can just open a PR, and it
will be reviewed, approved, and merged. This includes most bug fixes,
refactorings, and other user-invisible changes. The next few sections talk
about exceptions to this rule.

Also, note that it is perfectly acceptable to open WIP PRs or GitHub [Draft
PRs][draft]. Some people prefer to do this so they can get feedback along the
way or share their code with a collaborator. Others do this so they can utilize
the CI to build and test their PR (e.g. if you are developing on a laptop).

[draft]: https://github.blog/2019-02-14-introducing-draft-pull-requests/

### New Features

Rust has strong backwards-compatibility guarantees. Thus, new features can't
just be implemented directly in stable Rust. Instead, we have 3 release
channels: stable, beta, and nightly.

- **Stable**: this is the latest stable release for general usage.
- **Beta**: this is the next release (will be stable within 6 weeks).
- **Nightly**: follows the `master` branch of the repo. This is the only
  channel where unstable, incomplete, or experimental features are usable with
  feature gates.

In order to implement a new feature, usually you will need to go through [the
RFC process][rfc] to propose a design, have discussions, etc. In some cases,
small features can be added with only an FCP ([see below][break]). If in doubt, ask the
compiler, language, or libs team (whichever is most relevant).

[rfc]: https://github.com/rust-lang/rfcs/blob/master/README.md

After a feature is approved to be added, a tracking issue is created on the
`rust-lang/rust` repo, which tracks the progress towards the implementation of
the feature, any bugs reported, and eventually stabilization.

The feature then needs to be implemented behind a feature gate, which prevents
it from being accidentally used.

Finally, somebody may propose stabilizing the feature in an upcoming version of
Rust. This requires a Final Comment Period ([see below][break]) to get the
approval of the relevant teams.

After that, the feature gate can be removed and the feature turned on for all users.

For more details on this process, see [this chapter on implementing new
features.](./implementing_new_features.md)

### Breaking Changes

As mentioned above, Rust has strong backwards-compatibility guarantees. To this
end, we are reluctant to make breaking changes. However, sometimes they are
needed to correct compiler bugs (e.g. code that compiled but should not) or
make progress on some features.

Depending on the scale of the breakage, there are a few different actions that
can be taken.  If the reviewer believes the breakage is very minimal (i.e. very
unlikely to be actually encountered by users), they may just merge the change.
More often, they will request a Final Comment Period (FCP), which calls for
rough consensus among the members of a relevant team. The team members can
discuss the issue and either accept, reject, or request changes on the PR.

If the scale of breakage is large, a deprecation warning may be needed. This is
a warning that the compiler will display to users whose code will break in the
future. After some time, an FCP can be used to move forward with the actual
breakage.

If the scale of breakage is unknown, a team member or contributor may request a
[crater] run. This is a bot that will compile all crates.io crates and many
public github repos with the compiler with your changes. A report will then be
generated with crates that ceased to compile with or began to compile with your
changes. Crater runs can take a few days to complete.

[crater]: https://github.com/rust-lang/crater

### Major Changes

The compiler team has a special process for large changes, whether or not they
cause breakage. This process is called a Major Change Proposal (MCP). MCP is a
relatively lightweight mechanism for getting feedback on large changes to the
compiler (as opposed to a full RFC or a design meeting with the team).

Example of things that might require MCPs include major refactorings, changes
to important types, or important changes to how the compiler does something, or
smaller user-facing changes.

**When in doubt, ask on [zulip][z]. It would be a shame to put a lot of work
into a PR that ends up not getting merged!** [See this document][mcpinfo] for
more info on MCPs.

[mcpinfo]: https://forge.rust-lang.org/compiler/mcp.html

### Performance

Compiler performance is important. We have put a lot of effort over the last
few years into [gradually improving it][perfdash].

[perfdash]: https://perf.rust-lang.org/dashboard.html

If you suspect that your change may cause a performance regression (or
improvement), you can request a "perf run" (your reviewer may also request one
before approving). This is yet another bot that will compile a collection of
benchmarks on a compiler with your changes. The numbers are reported
[here][perf], and you can see a comparison of your changes against the latest
master.

[perf]: https://perf.rust-lang.org

## Other Resources

- This guide: talks about how `rustc` works
- [The t-compiler zulip][z]
- [The compiler's documentation (rustdocs)](https://doc.rust-lang.org/nightly/nightly-rustc/)
- [The Forge](https://forge.rust-lang.org/) has more documentation about various procedures.
- `#contribute` and `#rustdoc` on [Discord](https://discord.gg/rust-lang).
