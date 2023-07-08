# Contribution Procedures

<!-- toc -->

## Bug reports

While bugs are unfortunate, they're a reality in software. We can't fix what we
don't know about, so please report liberally. If you're not sure if something
is a bug or not, feel free to file a bug anyway.

**If you believe reporting your bug publicly represents a security risk to Rust users,
please follow our [instructions for reporting security vulnerabilities][vuln]**.

[vuln]: https://www.rust-lang.org/policies/security

If you're using the nightly channel, please check if the bug exists in the
latest toolchain before filing your bug. It might be fixed already.

If you have the chance, before reporting a bug, please [search existing issues],
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

## Bug fixes or "normal" code changes

For most PRs, no special procedures are needed. You can just [open a PR], and it
will be reviewed, approved, and merged. This includes most bug fixes,
refactorings, and other user-invisible changes. The next few sections talk
about exceptions to this rule.

Also, note that it is perfectly acceptable to open WIP PRs or GitHub [Draft PRs].
Some people prefer to do this so they can get feedback along the
way or share their code with a collaborator. Others do this so they can utilize
the CI to build and test their PR (e.g. when developing on a slow machine).

[open a PR]: #pull-requests
[Draft PRs]: https://github.blog/2019-02-14-introducing-draft-pull-requests/

## New features

Rust has strong backwards-compatibility guarantees. Thus, new features can't
just be implemented directly in stable Rust. Instead, we have 3 release
channels: stable, beta, and nightly.

- **Stable**: this is the latest stable release for general usage.
- **Beta**: this is the next release (will be stable within 6 weeks).
- **Nightly**: follows the `master` branch of the repo. This is the only
  channel where unstable, incomplete, or experimental features are usable with
  feature gates.

See [this chapter on implementing new features](./implementing_new_features.md) for more
information.

### Breaking changes

Breaking changes have a [dedicated section][Breaking Changes] in the dev-guide.

### Major changes

The compiler team has a special process for large changes, whether or not they
cause breakage. This process is called a Major Change Proposal (MCP). MCP is a
relatively lightweight mechanism for getting feedback on large changes to the
compiler (as opposed to a full RFC or a design meeting with the team).

Example of things that might require MCPs include major refactorings, changes
to important types, or important changes to how the compiler does something, or
smaller user-facing changes.

**When in doubt, ask on [zulip]. It would be a shame to put a lot of work
into a PR that ends up not getting merged!** [See this document][mcpinfo] for
more info on MCPs.

[mcpinfo]: https://forge.rust-lang.org/compiler/mcp.html
[zulip]: https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler

### Performance

Compiler performance is important. We have put a lot of effort over the last
few years into [gradually improving it][perfdash].

[perfdash]: https://perf.rust-lang.org/dashboard.html

If you suspect that your change may cause a performance regression (or
improvement), you can request a "perf run" (and your reviewer may also request one
before approving). This is yet another bot that will compile a collection of
benchmarks on a compiler with your changes. The numbers are reported
[here][perf], and you can see a comparison of your changes against the latest
master.

> For an introduction to the performance of Rust code in general
> which would also be useful in rustc development, see [The Rust Performance Book].

[perf]: https://perf.rust-lang.org
[The Rust Performance Book]: https://nnethercote.github.io/perf-book/

## Pull requests

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
As an example,
if you were making a diagnostics change,
then you could get a reviewer from the diagnostics team by adding:

    r? rust-lang/diagnostics

For a full list of possible `groupname`s,
check the `adhoc_groups` section at the [triagebot.toml config file],
or the list of teams in the [rust-lang teams database].

### Waiting for reviews

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

Please note that the reviewers are humans, who for the most part work on `rustc`
in their free time. This means that they can take some time to respond and review
your PR. It also means that reviewers can miss some PRs that are assigned to them.

To try to move PRs forward, the Triage WG regularly goes through all PRs that
are waiting for review and haven't been discussed for at least 2 weeks. If you
don't get a review within 2 weeks, feel free to ask the Triage WG on
Zulip ([#t-release/triage]). They have knowledge of when to ping, who might be
on vacation, etc.

The reviewer may request some changes using the GitHub code review interface.
They may also request special procedures for some PRs.
See [Crater] and [Breaking Changes] chapters for some examples of such procedures.

[r?]: https://github.com/rust-lang/rust/pull/78133#issuecomment-712692371
[#t-release/triage]: https://rust-lang.zulipchat.com/#narrow/stream/242269-t-release.2Ftriage
[Crater]: tests/crater.md

### CI

In addition to being reviewed by a human, pull requests are automatically tested,
thanks to continuous integration (CI). Basically, every time you open and update
a pull request, CI builds the compiler and tests it against the
[compiler test suite], and also performs other tests such as checking that
your pull request is in compliance with Rust's style guidelines.

Running continuous integration tests allows PR authors to catch mistakes early
without going through a first review cycle, and also helps reviewers stay aware
of the status of a particular pull request.

Rust has plenty of CI capacity, and you should never have to worry about wasting
computational resources each time you push a change. It is also perfectly fine
(and even encouraged!) to use the CI to test your changes if it can help your
productivity. In particular, we don't recommend running the full `./x test` suite locally,
since it takes a very long time to execute.

### r+

After someone has reviewed your pull request, they will leave an annotation
on the pull request with an `r+`. It will look something like this:

    @bors r+

This tells [@bors], our lovable integration bot, that your pull request has
been approved. The PR then enters the [merge queue], where [@bors]
will run *all* the tests on *every* platform we support. If it all works out,
[@bors] will merge your code into `master` and close the pull request.

Depending on the scale of the change, you may see a slightly different form of `r+`:

    @bors r+ rollup

The additional `rollup` tells [@bors] that this change should always be "rolled up".
Changes that are rolled up are tested and merged alongside other PRs, to
speed the process up. Typically only small changes that are expected not to conflict
with one another are marked as "always roll up".

Be patient; this can take a while and the queue can sometimes be long. PRs are never merged by hand.

[@rustbot]: https://github.com/rustbot
[@bors]: https://github.com/bors

### Opening a PR

You are now ready to file a pull request? Great! Here are a few points you
should be aware of.

All pull requests should be filed against the `master` branch,
unless you know for sure that you should target a different branch.

Make sure your pull request is in compliance with Rust's style guidelines by running

    $ ./x test tidy --bless

We recommend to make this check before every pull request (and every new commit
in a pull request); you can add [git hooks]
before every push to make sure you never forget to make this check.
The CI will also run tidy and will fail if tidy fails.

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

## External dependencies

This section has moved to ["Using External Repositories"](./external-repos.md).

## Writing documentation

Documentation improvements are very welcome. The source of `doc.rust-lang.org`
is located in [`src/doc`] in the tree, and standard API documentation is generated
from the source code itself (e.g. [`library/std/src/lib.rs`][std-root]). Documentation pull requests
function in the same way as other pull requests.

[`src/doc`]: https://github.com/rust-lang/rust/tree/master/src/doc
[std-root]: https://github.com/rust-lang/rust/blob/master/library/std/src/lib.rs#L1

To find documentation-related issues, sort by the [A-docs label].

You can find documentation style guidelines in [RFC 1574].

To build the standard library documentation, use `x doc --stage 0 library --open`.
To build the documentation for a book (e.g. the unstable book), use `x doc src/doc/unstable-book.`
Results should appear in `build/host/doc`, as well as automatically open in your default browser.
See [Building Documentation](./building/compiler-documenting.md#building-documentation) for more
information.

You can also use `rustdoc` directly to check small fixes. For example,
`rustdoc src/doc/reference.md` will render reference to `doc/reference.html`.
The CSS might be messed up, but you can verify that the HTML is right.

### Contributing to rustc-dev-guide

Contributions to the [rustc-dev-guide] are always welcome, and can be made directly at
[the rust-lang/rustc-dev-guide repo][rdgrepo].
The issue tracker in that repo is also a great way to find things that need doing.
There are issues for beginners and advanced compiler devs alike!

Just a few things to keep in mind:

- Please limit line length to 100 characters.
  This is enforced by CI,
  and you can run the checks locally with `ci/lengthcheck.sh`.

- When contributing text to the guide, please contextualize the information with some time period
  and/or a reason so that the reader knows how much to trust or mistrust the information.
  Aim to provide a reasonable amount of context, possibly including but not limited to:

  - A reason for why the data may be out of date other than "change",
    as change is a constant across the project.

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

## Issue triage

Sometimes, an issue will stay open, even though the bug has been fixed.
And sometimes, the original bug may go stale because something has changed in the meantime.

It can be helpful to go through older bug reports and make sure that they are still valid.
Load up an older issue, double check that it's still true,
and leave a comment letting us know if it is or is not.
The [least recently updated sort][lru] is good for finding issues like this.

[Thanks to `@rustbot`][rustbot], anyone can help triage issues by adding
appropriate labels to issues that haven't been triaged yet:

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

## Helpful links and information

This section has moved to the ["About this guide"] chapter.

["About this guide"]: about-this-guide.md#other-places-to-find-information
[search existing issues]: https://github.com/rust-lang/rust/issues?q=is%3Aissue
[Breaking Changes]: bug-fix-procedure.md
[triagebot.toml config file]: https://github.com/rust-lang/rust/blob/HEAD/triagebot.toml
[rust-lang teams database]: https://github.com/rust-lang/team/tree/HEAD/teams
[compiler test suite]: tests/intro.md
[merge queue]: https://bors.rust-lang.org/queue/rust
[git hooks]: https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
[A-docs label]: https://github.com/rust-lang/rust/issues?q=is%3Aopen%20is%3Aissue%20label%3AA-docs
[RFC 1574]: https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md#appendix-a-full-conventions-text
[rustc-dev-guide]: https://rustc-dev-guide.rust-lang.org/
[rdgrepo]: https://github.com/rust-lang/rustc-dev-guide
