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
bring those changes into the source repository. We have more info about how to use git
when contributing to Rust under [the git section](./git.md).

[about-pull-requests]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests
[development-models]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model

### r?

All pull requests are reviewed by another person. We have a bot,
[@rustbot], that will automatically assign a random person
to review your request based on which files you changed.

If you want to request that a specific person reviews your pull request, you
can add an `r?` to the pull request description or in a comment. For example,
if you want to ask a review to @awesome-reviewer, add

    r? @awesome-reviewer

to the end of the pull request description, and [@rustbot] will assign
them instead of a random person. This is entirely optional.

You can also assign a random reviewer from a specific team by writing `r? rust-lang/groupname`.
So if you were making a diagnostics change, then you could get a reviewer from the diagnostics
team by adding:

    r? rust-lang/diagnostics

For a full list of possible `groupname` check the `adhoc_groups` section at the
[triagebot.toml config file](https://github.com/rust-lang/rust/blob/master/triagebot.toml)
or the list of teams in the [rust-lang teams
database](https://github.com/rust-lang/team/tree/master/teams).

> NOTE
>
> Pull request reviewers are often working at capacity,
> and many of them are contributing on a volunteer basis.
> In order to minimize review delays,
> pull request authors and assigned reviewers should ensure that the review label
> (`S-waiting-on-review` and `S-waiting-on-author`) stays updated,
> invoking these commands when appropriate:
>
> - `@rustbot author`:
>   the review is finished,
>   and PR author should check the comments and take action accordingly.
>
> - `@rustbot review`:
>   the author is ready for a review,
>   and this PR will be queued again in the reviewer's queue.

### CI

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
productivity. In particular, we don't recommend running the full `./x.py test` suite locally,
since it takes a very long time to execute.

### r+

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

[@rustbot]: https://github.com/rustbot
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
branch.

If you encounter merge conflicts or when a reviewer asks you to perform some
changes, your PR will get marked as `S-waiting-on-author`. When you resolve
them, you should use `@rustbot` to mark it as `S-waiting-on-review`:

    @rustbot label -S-waiting-on-author +S-waiting-on-review

See [this chapter][labeling] for more details.

GitHub allows [closing issues using keywords][closing-keywords]. This feature
should be used to keep the issue tracker tidy. However, it is generally preferred
to put the "closes #123" text in the PR description rather than the issue commit;
particularly during rebasing, citing the issue number in the commit can "spam"
the issue in question.

[labeling]: ./rustbot.md#issue-relabeling
[closing-keywords]: https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue

### External Dependencies (subtree)

As a developer to this repository, you don't have to treat the following external projects
differently from other crates that are directly in this repo:

* [Clippy](https://github.com/rust-lang/rust-clippy)
* [Miri]
* [rustfmt](https://github.com/rust-lang/rustfmt)
* [rust-analyzer](https://github.com/rust-lang/rust-analyzer)

In contrast to `submodule` dependencies
(see below for those), the `subtree` dependencies are just regular files and directories which can
be updated in tree. However, if possible, enhancements, bug fixes, etc. specific
to these tools should be filed against the tools directly in their respective
upstream repositories. The exception is that when rustc changes are required to
implement a new tool feature or test, that should happen in one collective rustc PR.

#### Synchronizing a subtree

Periodically the changes made to subtree based dependencies need to be synchronized between this
repository and the upstream tool repositories.

Subtree synchronizations are typically handled by the respective tool maintainers. Other users
are welcome to submit synchronization PRs, however, in order to do so you will need to modify
your local git installation and follow a very precise set of instructions.
These instructions are documented, along with several useful tips and tricks, in the
[syncing subtree changes][clippy-sync-docs] section in Clippy's Contributing guide.
The instructions are applicable for use with any subtree based tool, just be sure to
use the correct corresponding subtree directory and remote repository.

The synchronization process goes in two directions: `subtree push` and `subtree pull`.

A `subtree push` takes all the changes that happened to the copy in this repo and creates commits
on the remote repo that match the local changes. Every local
commit that touched the subtree causes a commit on the remote repo, but
is modified to move the files from the specified directory to the tool repo root.

A `subtree pull` takes all changes since the last `subtree pull`
from the tool repo and adds these commits to the rustc repo along with a merge commit that moves
the tool changes into the specified directory in the Rust repository.

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

[clippy-sync-docs]: https://doc.rust-lang.org/nightly/clippy/development/infrastructure/sync.html

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

Building Rust will also use external git repositories tracked using [git
submodules]. The complete list may be found in the [`.gitmodules`] file. Some
of these projects are required (like `stdarch` for the standard library) and
some of them are optional (like [Miri]).

Usage of submodules is discussed more in the [Using Git
chapter](git.md#git-submodules).

Some of the submodules are allowed to be in a "broken" state where they
either don't build or their tests don't pass, e.g. the documentation books
like [The Rust Reference]. Maintainers of these projects will be notified
when the project is in a broken state, and they should fix them as soon
as possible. The current status is tracked on the [toolstate website].
More information may be found on the Forge [Toolstate chapter].

Breakage is not allowed in the beta and stable channels, and must be addressed
before the PR is merged. They are also not allowed to be broken on master in
the week leading up to the beta cut.

[git submodules]: https://git-scm.com/book/en/v2/Git-Tools-Submodules
[`.gitmodules`]: https://github.com/rust-lang/rust/blob/master/.gitmodules
[The Rust Reference]: https://github.com/rust-lang/reference/
[toolstate website]: https://rust-lang-nursery.github.io/rust-toolstate/
[Toolstate chapter]: https://forge.rust-lang.org/infra/toolstate.html

#### Breaking Tools Built With The Compiler

Rust's build system builds a number of tools that make use of the internals of
the compiler and that are hosted in a separate repository, and included in Rust
via git submodules (such as [Miri]). If these tools break because of your
changes, you may run into a sort of "chicken and egg" problem. These tools rely
on the latest compiler to be built so you can't update them (in their own
repositories) to reflect your changes to the compiler until those changes are
merged into the compiler. At the same time, you can't get your changes merged
into the compiler because the rust-lang/rust build won't pass until those tools
build and pass their tests.

Luckily, a feature was
[added to Rust's build](https://github.com/rust-lang/rust/issues/45861) to make
all of this easy to handle. The idea is that we allow these tools to be
"broken", so that the rust-lang/rust build passes without trying to build them,
then land the change in the compiler, and go update the tools that you
broke. Some tools will require waiting for a nightly release before this can
happen, while others use the builds uploaded after each bors merge and thus can
be updated immediately (check the tool's documentation for details). Once you're
done and the tools are working again, you go back in the compiler and update the
tools so they can be distributed again.

This should avoid a bunch of synchronization dances and is also much easier on contributors as
there's no need to block on tools changes going upstream.

Here are those same steps in detail:

1. (optional) First, if it doesn't exist already, create a `config.toml` by copying
   `config.example.toml` in the root directory of the Rust repository.
   Set `submodules = false` in the `[build]` section. This will prevent `x.py`
   from resetting to the original branch after you make your changes. If you
   need to [update any submodules to their latest versions](#updating-submodules),
   see the section of this file about that for more information.
2. (optional) Run `./x.py test src/tools/cargo` (substituting the submodule
   that broke for `cargo`). Fix any errors in the submodule (and possibly others).
3. (optional) Make commits for your changes and send them to upstream repositories as a PR.
4. (optional) Maintainers of these submodules will **not** merge the PR. The PR can't be
   merged because CI will be broken. You'll want to write a message on the PR referencing
   your change, and how the PR should be merged once your change makes it into a nightly.
5. Wait for your PR to merge.
6. Wait for a nightly.
7. (optional) Help land your PR on the upstream repository now that your changes are in nightly.
8. (optional) Send a PR to rust-lang/rust updating the submodule.


## Writing Documentation

Documentation improvements are very welcome. The source of `doc.rust-lang.org`
is located in [`src/doc`] in the tree, and standard API documentation is generated
from the source code itself (e.g. [`lib.rs`]). Documentation pull requests function
in the same way as other pull requests.

[`src/doc`]: https://github.com/rust-lang/rust/tree/master/src/doc
[`lib.rs`]: https://github.com/rust-lang/rust/blob/master/library/std/src/lib.rs#L1

To find documentation-related issues, sort by the [A-docs label][adocs].

[adocs]: https://github.com/rust-lang/rust/issues?q=is%3Aopen%20is%3Aissue%20label%3AA-docs

You can find documentation style guidelines in [RFC 1574][rfc1574].

[rfc1574]: https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md#appendix-a-full-conventions-text

In many cases, you don't need a full `./x.py doc --stage 2`, which will build
the entire stage 2 compiler and compile the various books published on
[doc.rust-lang.org][docs]. When updating documentation for the standard library,
first try `./x.py doc library`. If that fails, or if you need to
see the output from the latest version of `rustdoc`, add `--stage 1`.
Results should appear in `build/host/doc`.

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
    or _"As of now, ..."_,
    consider adding the date, in one of the following formats:
    - Jan 2021
    - January 2021
    - jan 2021
    - january 2021

    There is a CI action (in `~/.github/workflows/date-check.yml`)
    that generates a monthly issue with any of these that are over 6 months old.

    For the action to pick the date,
    add a special annotation before specifying the date:

    ```md
    <!-- date-check --> Jan 2023
    ```

    Example:

    ```md
    As of <!-- date-check --> Jan 2023, the foo did the bar.
    ```

    For cases where the date should not be part of the visible rendered output,
    use the following instead:

    ```md
    <!-- date-check: Jan 2023 -->
    ```

  - A link to a relevant WG, tracking issue, `rustc` rustdoc page, or similar, that may provide
    further explanation for the change process or a way to verify that the information is not
    outdated.

- If a text grows rather long (more than a few page scrolls) or complicated (more than four
  subsections) it might benefit from having a Table of Contents at the beginning, which you can
  auto-generate by including the `<!-- toc -->` marker.

[rdg]: https://rustc-dev-guide.rust-lang.org/
[rdgrepo]: https://github.com/rust-lang/rustc-dev-guide

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
  discussion at the next meeting of the team tagged using a
  **T**-prefixed label. Similarly, the [I-prioritize][ipri] indicates
  that an issue has been requested to be prioritized by the appropriate
  team.

* The purple **metabug** label marks lists of bugs collected by other
  categories.

* Purple gray, **O**-prefixed labels are the **operating system** or platform
  that this issue is specific to.

* Orange, **P**-prefixed labels indicate a bug's **priority**. These labels
  can be assigned by anyone that understand the issue and is able to
  prioritize it, and replace the [I-prioritize][ipri] label.

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

[rustc dev guide]: about-this-guide.md
[gdfrustc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/
[gsearchdocs]: https://www.google.com/search?q=site:doc.rust-lang.org+your+query+here
[stddocs]: https://doc.rust-lang.org/std
[rif]: http://internals.rust-lang.org
[rr]: https://doc.rust-lang.org/book/README.html
[rustforge]: https://forge.rust-lang.org/
[tlgba]: https://tomlee.co/2014/04/a-more-detailed-tour-of-the-rust-compiler/
[ro]: https://www.rustaceans.org/
[rctd]: tests/intro.md
[cheatsheet]: https://bors.rust-lang.org/
[Miri]: https://github.com/rust-lang/miri
