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
hop on the [Rust Discord server][rust-discord], [Rust Zulip server][rust-zulip] or [#rust-internals][pound-rust-internals].

As a reminder, all contributors are expected to follow our [Code of Conduct][coc].

The [rustc-guide] is your friend! It describes how the compiler works and how
to contribute to it in more detail than this document.

If this is your first time contributing, the [walkthrough] chapter of the guide
can give you a good example of how a typical contribution would go.

[pound-rust-internals]: https://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust-internals
[internals]: https://internals.rust-lang.org
[rust-discord]: http://discord.gg/rust-lang
[rust-zulip]: https://rust-lang.zulipchat.com
[coc]: https://www.rust-lang.org/conduct.html
[rustc-guide]: https://rust-lang.github.io/rustc-guide/
[walkthrough]: https://rust-lang.github.io/rustc-guide/walkthrough.html

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
please follow our [instructions for reporting security vulnerabilities](https://www.rust-lang.org/policies/security)**.

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

For info on how to configure and build the compiler, please see [this
chapter][rustcguidebuild] of the rustc-guide. This chapter contains info for
contributions to the compiler and the standard library. It also lists some
really useful commands to the build system (`./x.py`), which could save you a
lot of time.

[rustcguidebuild]: https://rust-lang.github.io/rustc-guide/how-to-build-and-run.html

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

Rust follows a no merge policy, meaning, when you encounter merge
conflicts you are expected to always rebase instead of merge.
E.g. always use rebase when bringing the latest changes from
the master branch to your feature branch.
Also, please make sure that fixup commits are squashed into other related
commits with meaningful commit messages.

Please make sure your pull request is in compliance with Rust's style
guidelines by running

    $ python x.py test src/tools/tidy

Make this check before every pull request (and every new commit in a pull
request); you can add [git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
before every push to make sure you never forget to make this check.

All pull requests are reviewed by another person. We have a bot,
[@rust-highfive][rust-highfive], that will automatically assign a random person to review your
request.

If you want to request that a specific person reviews your pull request,
you can add an `r?` to the message. For example, [Steve][steveklabnik] usually reviews
documentation changes. So if you were to make a documentation change, add

    r? @steveklabnik

to the end of the message, and @rust-highfive will assign [@steveklabnik][steveklabnik] instead
of a random person. This is entirely optional.

After someone has reviewed your pull request, they will leave an annotation
on the pull request with an `r+`. It will look something like this:

    @bors r+

This tells [@bors][bors], our lovable integration bot, that your pull request has
been approved. The PR then enters the [merge queue][merge-queue], where [@bors][bors]
will run all the tests on every platform we support. If it all works out,
[@bors][bors] will merge your code into `master` and close the pull request.

Depending on the scale of the change, you may see a slightly different form of `r+`:

    @bors r+ rollup

The additional `rollup` tells [@bors][bors] that this change is eligible for to be
"rolled up". Changes that are rolled up are tested and merged at the same time, to
speed the process up. Typically only small changes that are expected not to conflict
with one another are rolled up.

[rust-highfive]: https://github.com/rust-highfive
[steveklabnik]: https://github.com/steveklabnik
[bors]: https://github.com/bors
[merge-queue]: https://buildbot2.rust-lang.org/homu/queue/rust

Speaking of tests, Rust has a comprehensive test suite. More information about
it can be found [here][rctd].

### External Dependencies
[external-dependencies]: #external-dependencies

Currently building Rust will also build the following external projects:

* [clippy](https://github.com/rust-lang/rust-clippy)
* [miri](https://github.com/rust-lang/miri)
* [rustfmt](https://github.com/rust-lang/rustfmt)
* [rls](https://github.com/rust-lang/rls/)

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
internals of the compiler. This includes
[Clippy](https://github.com/rust-lang/rust-clippy),
[RLS](https://github.com/rust-lang/rls) and
[rustfmt](https://github.com/rust-lang/rustfmt). If these tools
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
error: failed to resolve patches for `https://github.com/rust-lang/rustfmt`

Caused by:
  patch for `rustfmt-nightly` in `https://github.com/rust-lang/rustfmt` did not resolve to any crates
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

```
$ cargo update -p rustfmt-nightly
```

This should change the version listed in `Cargo.lock` to the new version you updated
the submodule to. Running `./x.py build` should work now.

## Writing Documentation
[writing-documentation]: #writing-documentation

Documentation improvements are very welcome. The source of `doc.rust-lang.org`
is located in `src/doc` in the tree, and standard API documentation is generated
from the source code itself. Documentation pull requests function in the same way
as other pull requests.

To find documentation-related issues, sort by the [T-doc label][tdoc].

[tdoc]: https://github.com/rust-lang/rust/issues?q=is%3Aopen%20is%3Aissue%20label%3AT-doc

You can find documentation style guidelines in [RFC 1574][rfc1574].

[rfc1574]: https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md#appendix-a-full-conventions-text

In many cases, you don't need a full `./x.py doc`. You can use `rustdoc` directly
to check small fixes. For example, `rustdoc src/doc/reference.md` will render
reference to `doc/reference.html`. The CSS might be messed up, but you can
verify that the HTML is right.

Additionally, contributions to the [rustc-guide] are always welcome. Contributions
can be made directly at [the
rust-lang/rustc-guide](https://github.com/rust-lang/rustc-guide) repo. The issue
tracker in that repo is also a great way to find things that need doing. There
are issues for beginners and advanced compiler devs alike!

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
  RFC signoff functionality of [rfcbot][rfcbot] and are currently in the final
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

* The [rustc guide] contains information about how various parts of the compiler work and how to contribute to the compiler
* [Rust Forge][rustforge] contains additional documentation, including write-ups of how to achieve common tasks
* The [Rust Internals forum][rif], a place to ask questions and
  discuss Rust's internals
* The [generated documentation for rust's compiler][gdfrustc]
* The [rust reference][rr], even though it doesn't specifically talk about Rust's internals, it's a great resource nonetheless
* Although out of date, [Tom Lee's great blog article][tlgba] is very helpful
* [rustaceans.org][ro] is helpful, but mostly dedicated to IRC
* The [Rust Compiler Testing Docs][rctd]
* For [@bors][bors], [this cheat sheet][cheatsheet] is helpful
(though you'll need to replace `@homu` with `@bors` in any commands)
* **Google!** ([search only in Rust Documentation][gsearchdocs] to find types, traits, etc. quickly)
* Don't be afraid to ask! The Rust community is friendly and helpful.

[rustc guide]: https://rust-lang.github.io/rustc-guide/about-this-guide.html
[gdfrustc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/
[gsearchdocs]: https://www.google.com/search?q=site:doc.rust-lang.org+your+query+here
[rif]: http://internals.rust-lang.org
[rr]: https://doc.rust-lang.org/book/README.html
[rustforge]: https://forge.rust-lang.org/
[tlgba]: http://tomlee.co/2014/04/a-more-detailed-tour-of-the-rust-compiler/
[ro]: http://www.rustaceans.org/
[rctd]: https://rust-lang.github.io/rustc-guide/tests/intro.html
[cheatsheet]: https://buildbot2.rust-lang.org/homu/
