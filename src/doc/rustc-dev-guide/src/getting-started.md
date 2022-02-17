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

### System Requirements

Internet access is required.

The most notable software requirement is that you will need Python 2 or 3, but
there are various others.

The following hardware is recommended.
* 30GB+ of free disk space.
* 8GB+ RAM
* 2+ cores

More powerful machines will lead to much faster builds. There are various
strategies to work around lesser hardware in the following chapters.

See [this chapter][prereqs] for more details about software and hardware prerequisites.

[prereqs]: ./building/prerequisites.md

### Cloning

You can just do a normal git clone:

```sh
git clone https://github.com/rust-lang/rust.git
cd rust
```

### `x.py` Intro

`rustc` is a [bootstrapping] compiler, which makes it more complex than a
typical Rust program. As a result, you cannot use Cargo to build it. Instead
you must use the special tool `x.py`. It is used for the things Cargo is
normally used for: building, testing, creating releases, formatting, etc.

[bootstrapping]: ./building/bootstrapping.md

### Configuring the Compiler

In the top level of the repo:

```sh
$ ./x.py setup
```

This will do some initialization and walk you through an interactive setup to
create `config.toml`, the primary configuration file.

See [this chapter][config] for more info about configuration.

[config]: ./building/how-to-build-and-run.md#create-a-configtoml

### Common `x.py` commands

Here are the basic invocations of the `x.py` commands most commonly used when
working on `rustc`, `std`, `rustdoc`, and other tools.

| Command | When to use it |
| --- | --- |
| `./x.py check` | Quick check to see if most things compile; [rust-analyzer can run this automatically for you][rust-analyzer] |
| `./x.py build` | Builds `rustc`, `std`, and `rustdoc` |
| `./x.py test` | Runs all tests |
| `./x.py fmt` | Formats all code |

As written, these commands are reasonable starting points. However, there are
additional options and arguments for each of them that are worth learning for
serious development work. In particular, `./x.py build` and `./x.py test`
provide many ways to compile or test a subset of the code, which can save a lot
of time.

[rust-analyzer]: ./building/suggested.html#configuring-rust-analyzer-for-rustc

See the chapters on [building](./building/how-to-build-and-run.md),
[testing](./tests/intro.md), and [rustdoc](./rustdoc.md) for more details.

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
automatically assign a reviewer to the PR based on which files you changed.
The reviewer is the person that will approve the PR to be tested and merged.
If you want a specific reviewer (e.g. a team member you've been working with),
you can specifically request them by writing `r? @user` (e.g. `r? @jyn514`) in
either the original post or a followup comment
(you can see [this comment][r?] for example).

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

For an introduction to the performance of Rust code in general
which would also be useful in rustc development, see [The Rust Performance Book].

[perf]: https://perf.rust-lang.org
[The Rust Performance Book]: https://nnethercote.github.io/perf-book/

## Other Resources

- This guide: talks about how `rustc` works
- [The t-compiler zulip][z]
- [The compiler's documentation (rustdocs)](https://doc.rust-lang.org/nightly/nightly-rustc/)
- [The Forge](https://forge.rust-lang.org/) has more documentation about various procedures.
- `#contribute` and `#rustdoc` on [Discord](https://discord.gg/rust-lang).
