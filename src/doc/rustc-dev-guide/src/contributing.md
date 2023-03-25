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

## External Dependencies

This sections has moved to ["Using External Repositories"](./external-repos.md).

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

If you're looking for somewhere to start, check out the [E-easy] tag.

[Thanks to `@rustbot`][rustbot], anyone can help triage issues by adding
appropriate labels to issues that haven't been triaged yet:

[E-easy]: https://github.com/rust-lang/rust/issues?q=is%3Aopen+is%3Aissue+label%3AE-easy
[lru]: https://github.com/rust-lang/rust/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-asc
[rustbot]: ./rustbot.md

<style>
.label-color {
    border-radius:0.5em;
}
table td:nth-child(2) {
    white-space: nowrap;
}

</style>

| Labels | Color | Description |
|--------|-------|-------------|
| [A-]   | <span class="label-color" style="background-color:#f7e101;">&#x2003;</span>&nbsp;Yellow | The **area** of the project an issue relates to. |
| [B-]   | <span class="label-color" style="background-color:#d304cb;">&#x2003;</span>&nbsp;Magenta | Issues which are **blockers**. |
| [beta-] | <span class="label-color" style="background-color:#1e76d9;">&#x2003;</span>&nbsp;Dark Blue | Tracks changes which need to be [backported to beta][beta-backport] |
| [C-] | <span class="label-color" style="background-color:#f5f1fd;">&#x2003;</span>&nbsp;Light Purple | The **category** of an issue. |
| [D-] | <span class="label-color" style="background-color:#c9f7a3;">&#x2003;</span>&nbsp;Mossy Green | Issues for **diagnostics**. |
| [E-] | <span class="label-color" style="background-color:#02e10c;">&#x2003;</span>&nbsp;Green | The **experience** level necessary to fix an issue. |
| [F-] | <span class="label-color" style="background-color:#f9c0cc;">&#x2003;</span>&nbsp;Peach | Issues for **nightly features**. |
| [I-] | <span class="label-color" style="background-color:#e10c02;">&#x2003;</span>&nbsp;Red | The **importance** of the issue. |
| [I-\*-nominated] | <span class="label-color" style="background-color:#e10c02;">&#x2003;</span>&nbsp;Red | The issue has been nominated for discussion at the next meeting of the corresponding team. |
| [I-prioritize] | <span class="label-color" style="background-color:#e10c02;">&#x2003;</span>&nbsp;Red | The issue has been nominated for prioritization by the team tagged with a **T**-prefixed label. |
| [metabug] | <span class="label-color" style="background-color:#5319e7;">&#x2003;</span>&nbsp;Purple | Bugs that collect other bugs. |
| [O-] | <span class="label-color" style="background-color:#6e6ec0;">&#x2003;</span>&nbsp;Purple Grey | The **operating system** or platform that the issue is specific to. |
| [P-] | <span class="label-color" style="background-color:#eb6420;">&#x2003;</span>&nbsp;Orange | The issue **priority**.  These labels can be assigned by anyone that understand the issue and is able to  prioritize it, and remove the [I-prioritize] label. |
| [regression-] | <span class="label-color" style="background-color:#e4008a;">&#x2003;</span>&nbsp;Pink | Tracks regressions from a stable release. |
| [relnotes] | <span class="label-color" style="background-color:#fad8c7;">&#x2003;</span>&nbsp;Light Orange | Changes that should be documented in the release notes of the next release. |
| [S-] | <span class="label-color" style="background-color:#d3dddd;">&#x2003;</span>&nbsp;Gray | Tracks the **status** of pull requests. |
| [S-tracking-] | <span class="label-color" style="background-color:#4682b4;">&#x2003;</span>&nbsp;Steel Blue | Tracks the **status** of [tracking issues]. |
| [stable-] | <span class="label-color" style="background-color:#00229c;">&#x2003;</span>&nbsp;Dark Blue | Tracks changes which need to be [backported to stable][stable-backport] in anticipation of a point release. |
| [T-] | <span class="label-color" style="background-color:#bfd4f2;">&#x2003;</span>&nbsp;Blue | Denotes which **team** the issue belongs to. |
| [WG-] | <span class="label-color" style="background-color:#c2e0c6;">&#x2003;</span>&nbsp;Green | Denotes which **working group** the issue belongs to. |


[A-]: https://github.com/rust-lang/rust/labels?q=A
[B-]: https://github.com/rust-lang/rust/labels?q=B
[C-]: https://github.com/rust-lang/rust/labels?q=C
[D-]: https://github.com/rust-lang/rust/labels?q=D
[E-]: https://github.com/rust-lang/rust/labels?q=E
[F-]: https://github.com/rust-lang/rust/labels?q=F
[I-]: https://github.com/rust-lang/rust/labels?q=I
[O-]: https://github.com/rust-lang/rust/labels?q=O
[P-]: https://github.com/rust-lang/rust/labels?q=P
[S-]: https://github.com/rust-lang/rust/labels?q=S
[T-]: https://github.com/rust-lang/rust/labels?q=T
[WG-]: https://github.com/rust-lang/rust/labels?q=WG
[stable-]: https://github.com/rust-lang/rust/labels?q=stable
[beta-]: https://github.com/rust-lang/rust/labels?q=beta
[I-\*-nominated]: https://github.com/rust-lang/rust/labels?q=nominated
[I-prioritize]: https://github.com/rust-lang/rust/labels/I-prioritize
[tracking issues]: https://github.com/rust-lang/rust/labels/C-tracking-issue
[beta-backport]: https://forge.rust-lang.org/release/backporting.html#beta-backporting-in-rust-langrust
[stable-backport]: https://forge.rust-lang.org/release/backporting.html#stable-backporting-in-rust-langrust
[metabug]: https://github.com/rust-lang/rust/labels/metabug
[regression-]: https://github.com/rust-lang/rust/labels?q=regression
[relnotes]: https://github.com/rust-lang/rust/labels/relnotes
[S-tracking-]: https://github.com/rust-lang/rust/labels?q=s-tracking

### Rfcbot labels

[rfcbot] uses its own labels for tracking the process of coordinating
asynchronous decisions, such as approving or rejecting a change.
This is used for [RFCs], issues, and pull requests.

| Labels | Color | Description |
|--------|-------|-------------|
| [proposed-final-comment-period] | <span class="label-color" style="background-color:#ededed;">&#x2003;</span>&nbsp;Gray | Currently awaiting signoff of all team members in order to enter the final comment period. |
| [disposition-merge] | <span class="label-color" style="background-color:#008800;">&#x2003;</span>&nbsp;Green | Indicates the intent is to merge the change. |
| [disposition-close] | <span class="label-color" style="background-color:#dd0000;">&#x2003;</span>&nbsp;Red | Indicates the intent is to not accept the change and close it. |
| [disposition-postpone] | <span class="label-color" style="background-color:#ededed;">&#x2003;</span>&nbsp;Gray | Indicates the intent is to not accept the change at this time and postpone it to a later date. |
| [final-comment-period] | <span class="label-color" style="background-color:#1e76d9;">&#x2003;</span>&nbsp;Blue | Currently soliciting final comments before merging or closing. |
| [finished-final-comment-period] | <span class="label-color" style="background-color:#f9e189;">&#x2003;</span>&nbsp;Light Yellow | The final comment period has concluded, and the issue will be merged or closed. |
| [postponed] | <span class="label-color" style="background-color:#fbca04;">&#x2003;</span>&nbsp;Yellow | The issue has been postponed. |
| [closed] | <span class="label-color" style="background-color:#dd0000;">&#x2003;</span>&nbsp;Red | The issue has been rejected. |
| [to-announce] | <span class="label-color" style="background-color:#ededed;">&#x2003;</span>&nbsp;Gray | Issues that have finished their final-comment-period and should be publicly announced. Note: the rust-lang/rust repository uses this label differently, to announce issues at the triage meetings. |

[disposition-merge]: https://github.com/rust-lang/rust/labels/disposition-merge
[disposition-close]: https://github.com/rust-lang/rust/labels/disposition-close
[disposition-postpone]: https://github.com/rust-lang/rust/labels/disposition-postpone
[proposed-final-comment-period]: https://github.com/rust-lang/rust/labels/proposed-final-comment-period
[final-comment-period]: https://github.com/rust-lang/rust/labels/final-comment-period
[finished-final-comment-period]: https://github.com/rust-lang/rust/labels/finished-final-comment-period
[postponed]: https://github.com/rust-lang/rfcs/labels/postponed
[closed]: https://github.com/rust-lang/rfcs/labels/closed
[to-announce]: https://github.com/rust-lang/rfcs/labels/to-announce
[rfcbot]: https://github.com/anp/rfcbot-rs/
[RFCs]: https://github.com/rust-lang/rfcs

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
