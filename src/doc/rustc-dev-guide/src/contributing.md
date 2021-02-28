# Contributing to Rust

Thank you for your interest in contributing to Rust! There are many ways to
contribute, and we appreciate all of them.

<!-- toc -->

If you have questions, please make a post on [internals.rust-lang.org][internals] or
hop on the [Rust Discord server][rust-discord] or [Rust Zulip server][rust-zulip].

As a reminder, all contributors are expected to follow our [Code of Conduct][coc].

If this is your first time contributing, the [Getting Started] and
[walkthrough] chapters can give you a good example of how a typical
contribution would go.

[internals]: https://internals.rust-lang.org
[rust-discord]: http://discord.gg/rust-lang
[rust-zulip]: https://rust-lang.zulipchat.com
[coc]: https://www.rust-lang.org/conduct.html
[walkthrough]: ./walkthrough.md
[Getting Started]: ./getting-started.md

## Feature Requests

Feature requests need to go through a process to be approved by the relevant
teams. Usually this requires a Final Comment Period (FCP) or even a Request for
Comments (RFC). See [Getting Started] for more information about these processes.

## Bug Reports

While bugs are unfortunate, they're a reality in software. We can't fix what we
don't know about, so please report liberally. If you're not sure if something
is a bug or not, feel free to file a bug anyway.

**If you believe reporting your bug publicly represents a security risk to Rust users,
please follow our [instructions for reporting security vulnerabilities][vuln]**.

[vuln]: https://www.rust-lang.org/policies/security

If you're using the nightly channel, please check if the bug exists in the
latest toolchain before filing your bug. It might be fixed already.

If you have the chance, before reporting a bug, please [search existing
issues](https://github.com/rust-lang/rust/issues?q=is%3Aissue),
as it's possible that someone else has already reported your error. This doesn't
always work, and sometimes it's hard to know what to search for, so consider this
extra credit. We won't mind if you accidentally file a duplicate report.

Similarly, to help others who encountered the bug find your issue, consider
filing an issue with a descriptive title, which contains information that might
be unique to it.  This can be the language or compiler feature used, the
conditions that trigger the bug, or part of the error message if there is any.
An example could be: **"impossible case reached" on lifetime inference for impl
Trait in return position**.

Opening an issue is as easy as following [this
link](https://github.com/rust-lang/rust/issues/new/choose) and filling out the fields
in the appropriate provided template.

## Pull Requests

Pull requests (or PRs for short) are the primary mechanism we use to change Rust.
GitHub itself has some [great documentation][about-pull-requests] on using the
Pull Request feature. We use the "fork and pull" model [described here][development-models],
where contributors push changes to their personal fork and create pull requests to
bring those changes into the source repository.

[about-pull-requests]: https://help.github.com/articles/about-pull-requests/
[development-models]: https://help.github.com/articles/about-collaborative-development-models/

All pull requests are reviewed by another person. We have a bot,
[@rust-highfive][rust-highfive], that will automatically assign a random person
to review your request.

If you want to request that a specific person reviews your pull request, you
can add an `r?` to the pull request description. For example,
[Steve][steveklabnik] usually reviews documentation changes. So if you were to
make a documentation change, add

    r? @steveklabnik

to the end of the pull request description, and [@rust-highfive][rust-highfive] will assign
[@steveklabnik][steveklabnik] instead of a random person. This is entirely optional.

In addition to being reviewed by a human, pull requests are automatically tested
thanks to continuous integration (CI). Basically, every time you open and update
a pull request, CI builds the compiler and tests it against the
[compiler test suite][rctd], and also performs other tests such as checking that
your pull request is in compliance with Rust's style guidelines.

Running continuous integration tests allows PR authors to catch mistakes early
without going through a first review cycle, and also helps reviewers stay aware
of the status of a particular pull request.

Rust has plenty of CI capacity, and you should never have to worry about wasting
computational resources each time you push a change. It is also perfectly fine
(and even encouraged!) to use the CI to test your changes if it can help your
productivity. In particular, we don't recommend running the full `x.py test` suite locally,
since it takes a very long time to execute.

After someone has reviewed your pull request, they will leave an annotation
on the pull request with an `r+`. It will look something like this:

    @bors r+

This tells [@bors], our lovable integration bot, that your pull request has
been approved. The PR then enters the [merge queue][merge-queue], where [@bors]
will run *all* the tests on *every* platform we support. If it all works out,
[@bors] will merge your code into `master` and close the pull request.

Depending on the scale of the change, you may see a slightly different form of `r+`:

    @bors r+ rollup

The additional `rollup` tells [@bors] that this change should always be "rolled up".
Changes that are rolled up are tested and merged alongside other PRs, to
speed the process up. Typically only small changes that are expected not to conflict
with one another are marked as "always roll up".

[rust-highfive]: https://github.com/rust-highfive
[steveklabnik]: https://github.com/steveklabnik
[@bors]: https://github.com/bors
[merge-queue]: https://bors.rust-lang.org/queue/rust

### Opening a PR

You are now ready to file a pull request? Great! Here are a few points you
should be aware of.

All pull requests should be filed against the `master` branch, except in very
particular scenarios. Unless you know for sure that you should target another
branch, `master` will be the right choice (it's also the default).

Make sure your pull request is in compliance with Rust's style guidelines by running

    $ ./x.py test tidy --bless

We recommend to make this check before every pull request (and every new commit
in a pull request); you can add [git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
before every push to make sure you never forget to make this check. The
CI will also run tidy and will fail if tidy fails.

Rust follows a _no merge-commit policy_, meaning, when you encounter merge
conflicts you are expected to always rebase instead of merging.  E.g. always use
rebase when bringing the latest changes from the master branch to your feature
branch.  Also, please make sure that fixup commits are squashed into other
related commits with meaningful commit messages.

If you encounter merge conflicts, your PR will get marked as `S-waiting-on-author`.
When you resolve them, you should use `@rustbot` to mark it as `S-waiting-on-review`.
See [this chapter][labeling] for more details.

GitHub allows [closing issues using keywords][closing-keywords]. This feature
should be used to keep the issue tracker tidy. However, it is generally preferred
to put the "closes #123" text in the PR description rather than the issue commit;
particularly during rebasing, citing the issue number in the commit can "spam"
the issue in question.

[labeling]: ./rustbot.md#issue-relabeling
[closing-keywords]: https://help.github.com/en/articles/closing-issues-using-keywords

### External Dependencies (subtree)

As a developer to this repository, you don't have to treat the following external projects
differently from other crates that are directly in this repo:

* Clippy

They are just regular files and directories. This is in contrast to `submodule` dependencies
(see below for those). Only tool authors will actually use any operations here.

#### Synchronizing a subtree

There are two synchronization directions: `subtree push` and `subtree pull`.

```
git subtree push -P src/tools/clippy git@github.com:your-github-name/rust-clippy sync-from-rust
```

takes all the changes that
happened to the copy in this repo and creates commits on the remote repo that match the local
changes. Every local commit that touched the subtree causes a commit on the remote repo, but is
modified to move the files from the specified directory to the tool repo root.

Make sure to not pick the `master` branch on the tool repo, so you can open a normal PR to the tool
to merge that subrepo push.

```
git subtree pull -P src/tools/clippy https://github.com/rust-lang/rust-clippy master
```

takes all changes since the last `subtree pull` from the tool repo
repo and adds these commits to the rustc repo + a merge commit that moves the tool changes into
the specified directory in the rust repository.

It is recommended that you always do a push first and get that merged to the tool master branch.
Then, when you do a pull, the merge works without conflicts.
While it's definitely possible to resolve conflicts during a pull, you may have to redo the conflict
resolution if your PR doesn't get merged fast enough and there are new conflicts. Do not try to
rebase the result of a `git subtree pull`, rebasing merge commits is a bad idea in general.

You always need to specify the `-P` prefix to the subtree directory and the corresponding remote
repository. If you specify the wrong directory or repository
you'll get very fun merges that try to push the wrong directory to the wrong remote repository.
Luckily you can just abort this without any consequences by throwing away either the pulled commits
in rustc or the pushed branch on the remote and try again. It is usually fairly obvious
that this is happening because you suddenly get thousands of commits that want to be synchronized.

#### Creating a new subtree dependency

If you want to create a new subtree dependency from an existing repository, call (from this
repository's root directory!)

```
git subtree add -P src/tools/clippy https://github.com/rust-lang/rust-clippy.git master
```

This will create a new commit, which you may not rebase under any circumstances! Delete the commit
and redo the operation if you need to rebase.

Now you're done, the `src/tools/clippy` directory behaves as if Clippy were
part of the rustc monorepo, so no one but you (or others that synchronize
subtrees) actually needs to use `git subtree`.


### External Dependencies (submodules)

Currently building Rust will also build the following external projects:

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

Rust's build system builds a number of tools that make use of the
internals of the compiler. This includes
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
   need to [update any submodules to their latest versions](#updating-submodules),
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

The [`[patch]`][patchsec] section of `Cargo.toml` can be very useful for
testing. In addition to that, you should read the [Overriding
dependencies][overriding] section of the documentation.

[patchsec]: http://doc.crates.io/manifest.html#the-patch-section
[overriding]: http://doc.crates.io/specifying-dependencies.html#overriding-dependencies

Specifically, the following [section in Overriding dependencies][testingbugfix]
reveals what the problem is:

[testingbugfix]: http://doc.crates.io/specifying-dependencies.html#testing-a-bugfix

> Next up we need to ensure that our lock file is updated to use this new
> version of uuid so our project uses the locally checked out copy instead of
> one from crates.io. The way `[patch]` works is that it'll load the dependency
> at ../path/to/uuid and then whenever crates.io is queried for versions of
> uuid it'll also return the local version.
>
> This means that the version number of the local checkout is significant and
> will affect whether the patch is used. Our manifest declared uuid = "1.0"
> which means we'll only resolve to >= 1.0.0, < 2.0.0, and Cargo's greedy
> resolution algorithm also means that we'll resolve to the maximum version
> within that range. Typically this doesn't matter as the version of the git
> repository will already be greater or match the maximum version published on
> crates.io, but it's important to keep this in mind!

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

Documentation improvements are very welcome. The source of `doc.rust-lang.org`
is located in [`src/doc`] in the tree, and standard API documentation is generated
from the source code itself (e.g. [`lib.rs`]). Documentation pull requests function
in the same way as other pull requests.

[`src/doc`]: https://github.com/rust-lang/rust/tree/master/src/doc
[`lib.rs`]: https://github.com/rust-lang/rust/blob/master/library/std/src/lib.rs#L1

To find documentation-related issues, sort by the [T-doc label][tdoc].

[tdoc]: https://github.com/rust-lang/rust/issues?q=is%3Aopen%20is%3Aissue%20label%3AT-doc

You can find documentation style guidelines in [RFC 1574][rfc1574].

[rfc1574]: https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md#appendix-a-full-conventions-text

In many cases, you don't need a full `./x.py doc --stage 2`, which will build
the entire stage 2 compiler and compile the various books published on
[doc.rust-lang.org][docs]. When updating documentation for the standard library,
first try `./x.py doc library/std`. If that fails, or if you need to
see the output from the latest version of `rustdoc`, add `--stage 1`.
Results should appear in `build/$TARGET/doc`.

[docs]: https://doc.rust-lang.org

You can also use `rustdoc` directly to check small fixes. For example,
`rustdoc src/doc/reference.md` will render reference to `doc/reference.html`.
The CSS might be messed up, but you can verify that the HTML is right.

### Contributing to rustc-dev-guide

Contributions to the [rustc-dev-guide][rdg] are always welcome, and can be made directly at
[the rust-lang/rustc-dev-guide repo][rdgrepo].
The issue tracker in that repo is also a great way to find things that need doing.
There are issues for beginners and advanced compiler devs alike!

Just a few things to keep in mind:

- Please limit line length to 100 characters. This is enforced by CI, and you can run the checks
  locally with `ci/check_line_lengths.sh`.

- When contributing text to the guide, please contextualize the information with some time period
  and/or a reason so that the reader knows how much to trust or mistrust the information.
  Aim to provide a reasonable amount of context, possibly including but not limited to:

  - A reason for why the data may be out of date other than "change", as change is a constant across
    the project.

  - The date the comment was added, e.g. instead of writing _"Currently, ..."_
    or _"As of now, ..."_, consider writing
    _"As of January 2021, ..."_.
    Try to format the date as `<MONTH> <YEAR>` to ease search.

  - Additionally, include a machine-readable comment of the form `<!-- date:
    2021-01 -->` (if the current month is January 2021). We have an automated
    tool that uses these (in `ci/date-check`).

    So, for the month of January 2021, the comment would look like: `As of <!--
    date: 2021-01 --> January 2021`. Make sure to put the comment *between* `as of`
    and `January 2021`; see [PR #1066][rdg#1066] for the rationale.

  - A link to a relevant WG, tracking issue, `rustc` rustdoc page, or similar, that may provide
    further explanation for the change process or a way to verify that the information is not
    outdated.

- If a text grows rather long (more than a few page scrolls) or complicated (more than four
  subsections) it might benefit from having a Table of Contents at the beginning, which you can
  auto-generate by including the `<!-- toc -->` marker.

[rdg]: https://rustc-dev-guide.rust-lang.org/
[rdgrepo]: https://github.com/rust-lang/rustc-dev-guide
[rdg#1066]: https://github.com/rust-lang/rustc-dev-guide/pull/1066

## Issue Triage

Sometimes, an issue will stay open, even though the bug has been fixed. And
sometimes, the original bug may go stale because something has changed in the
meantime.

It can be helpful to go through older bug reports and make sure that they are
still valid. Load up an older issue, double check that it's still true, and
leave a comment letting us know if it is or is not. The [least recently
updated sort][lru] is good for finding issues like this.

[Thanks to `@rustbot`][rustbot], anyone can help triage issues by adding
appropriate labels to issues that haven't been triaged yet:

* Yellow, **A**-prefixed labels state which **area** of the project an issue
  relates to.

* Magenta, **B**-prefixed labels identify bugs which are **blockers**.

* Dark blue, **beta-** labels track changes which need to be backported into
  the beta branches.

* Light purple, **C**-prefixed labels represent the **category** of an issue.

* Green, **E**-prefixed labels explain the level of **experience** necessary
  to fix the issue.

* The dark blue **final-comment-period** label marks bugs that are using the
  RFC signoff functionality of [rfcbot] and are currently in the final
  comment period.

* Red, **I**-prefixed labels indicate the **importance** of the issue. The
  [I-nominated][inom] label indicates that an issue has been nominated for
  prioritizing at the next triage meeting. Similarly, the [I-prioritize][ipri]
  indicates that an issue has been requested to be prioritized by the
  appropriate team.

* The purple **metabug** label marks lists of bugs collected by other
  categories.

* Purple gray, **O**-prefixed labels are the **operating system** or platform
  that this issue is specific to.

* Orange, **P**-prefixed labels indicate a bug's **priority**. These labels
  are only assigned during triage meetings, and replace the [I-prioritize][ipri]
  label.

* The gray **proposed-final-comment-period** label marks bugs that are using
  the RFC signoff functionality of [rfcbot] and are currently awaiting
  signoff of all team members in order to enter the final comment period.

* Pink, **regression**-prefixed labels track regressions from stable to the
  release channels.

* The light orange **relnotes** label marks issues that should be documented in
  the release notes of the next release.

* Gray, **S**-prefixed labels are used for tracking the **status** of pull
  requests.

* Blue, **T**-prefixed bugs denote which **team** the issue belongs to.

If you're looking for somewhere to start, check out the [E-easy][eeasy] tag.

[rustbot]: ./rustbot.md
[inom]: https://github.com/rust-lang/rust/issues?q=is%3Aopen+is%3Aissue+label%3AI-nominated
[ipri]: https://github.com/rust-lang/rust/issues?q=is%3Aopen+is%3Aissue+label%3AI-prioritize
[eeasy]: https://github.com/rust-lang/rust/issues?q=is%3Aopen+is%3Aissue+label%3AE-easy
[lru]: https://github.com/rust-lang/rust/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-asc
[rfcbot]: https://github.com/anp/rfcbot-rs/

## Out-of-tree Contributions

There are a number of other ways to contribute to Rust that don't deal with
rust-lang/rust:

* Answer questions in the _Get Help!_ channels on the [Rust Discord
  server][rust-discord], on [users.rust-lang.org][users], or on
  [StackOverflow][so].
* Participate in the [RFC process](https://github.com/rust-lang/rfcs).
* Find a [requested community library][community-library], build it, and publish
  it to [Crates.io](http://crates.io). Easier said than done, but very, very
  valuable!

[rust-discord]: https://discord.gg/rust-lang
[users]: https://users.rust-lang.org/
[so]: http://stackoverflow.com/questions/tagged/rust
[community-library]: https://github.com/rust-lang/rfcs/labels/A-community-library

## Helpful Links and Information

For people new to Rust, and just starting to contribute, or even for
more seasoned developers, some useful places to look for information
are:

* This guide contains information about how various parts of the
  compiler work and how to contribute to the compiler
* [Rust Forge][rustforge] contains additional documentation, including
  write-ups of how to achieve common tasks
* The [Rust Internals forum][rif], a place to ask questions and
  discuss Rust's internals
* The [generated documentation for Rust's compiler][gdfrustc]
* The [Rust reference][rr], even though it doesn't specifically talk about
  Rust's internals, is a great resource nonetheless
* Although out of date, [Tom Lee's great blog article][tlgba] is very helpful
* [rustaceans.org][ro] is helpful, but mostly dedicated to IRC
* The [Rust Compiler Testing Docs][rctd]
* For [@bors], [this cheat sheet][cheatsheet] is helpful
* Google is always helpful when programming.
  You can [search all Rust documentation][gsearchdocs] (the standard library,
  the compiler, the books, the references, and the guides) to quickly find
  information about the language and compiler.
* You can also use Rustdoc's built-in search feature to find documentation on
  types and functions within the crates you're looking at. You can also search
  by type signature! For example, searching for `* -> vec` should find all
  functions that return a `Vec<T>`.
  _Hint:_ Find more tips and keyboard shortcuts by typing `?` on any Rustdoc
  page!
* Don't be afraid to ask! The Rust community is friendly and helpful.

[rustc dev guide]: https://rustc-dev-guide.rust-lang.org/about-this-guide.html
[gdfrustc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/
[gsearchdocs]: https://www.google.com/search?q=site:doc.rust-lang.org+your+query+here
[stddocs]: https://doc.rust-lang.org/std
[rif]: http://internals.rust-lang.org
[rr]: https://doc.rust-lang.org/book/README.html
[rustforge]: https://forge.rust-lang.org/
[tlgba]: https://tomlee.co/2014/04/a-more-detailed-tour-of-the-rust-compiler/
[ro]: https://www.rustaceans.org/
[rctd]: https://rustc-dev-guide.rust-lang.org/tests/intro.html
[cheatsheet]: https://bors.rust-lang.org/
