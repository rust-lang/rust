# Contribution procedures

## Bug reports

While bugs are unfortunate, they're a reality in software.
We can't fix what we don't know about, so please report liberally.
If you're not sure if something is a bug, feel free to open an issue anyway.

**If you believe reporting your bug publicly represents a security risk to Rust users,
please follow our [instructions for reporting security vulnerabilities][vuln]**.

[vuln]: https://www.rust-lang.org/policies/security

If you're using the nightly channel, please check if the bug exists in the
latest toolchain before filing your bug.
It might be fixed already.

If you have the chance, before reporting a bug, please [search existing issues],
as it's possible that someone else has already reported your error.
This doesn't always work, and sometimes it's hard to know what to search for, so consider this
extra credit.
We won't mind if you accidentally file a duplicate report.

Similarly, to help others who encountered the bug find your issue, consider
filing an issue with a descriptive title, which contains information that might be unique to it.
This can be the language or compiler feature used, the
conditions that trigger the bug, or part of the error message if there is any.
An example could be: **"impossible case reached" on lifetime inference for impl
Trait in return position**.

Opening an issue is as easy as following [this link][create an issue] and filling out the fields
in the appropriate provided template.

## New features

Rust has strong backwards-compatibility guarantees.
Thus, new features can't just be implemented directly in stable Rust.
Instead, we have 3 release channels: stable, beta, and nightly.
See [The Rust Book] for more details on Rust’s train release model.

- **Stable**: this is the latest stable release for general usage.
- **Beta**: this is the next release (will be stable within 6 weeks).
- **Nightly**: follows the `main` branch of the repo.
  This is the only channel where unstable features are intended to be used,
  which happens via opt-in feature gates.

See [this chapter on implementing new features](./implementing-new-features.md) for more
information.

[The Rust Book]: https://doc.rust-lang.org/book/appendix-07-nightly-rust.html

### Breaking changes

Breaking changes have a [dedicated section][Breaking Changes] in the dev-guide.

### Major changes

See ["What proposal approval do I need?"](https://forge.rust-lang.org/compiler/proposals-and-stabilization.html#what-proposalapproval-do-i-need)
For a definition of the terms there, see
["Proposals"](https://forge.rust-lang.org/compiler/proposals-and-stabilization.html#proposals).

**When in doubt, ask [on Zulip].
It would be a shame to put a lot of work into a PR that ends up not getting merged!**

[on Zulip]: https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler

### Performance

Compiler performance is important.
We have put a lot of effort over the last few years into [gradually improving it][perfdash].

[perfdash]: https://perf.rust-lang.org/dashboard.html

If you suspect that your change may cause a performance regression (or
improvement), you can request a "perf run" (and your reviewer may also request one
before approving).
This is yet another bot that will compile a collection of
benchmarks on a compiler with your changes.
The numbers are reported
[here][perf], and you can see a comparison of your changes against the latest `main`.

> For an introduction to the performance of Rust code in general
> which would also be useful in rustc development, see [The Rust Performance Book].

[perf]: https://perf.rust-lang.org
[The Rust Performance Book]: https://nnethercote.github.io/perf-book/

### Keeping your branch up-to-date

The CI in rust-lang/rust applies your patches directly against current `main`,
not against the commit your branch is based on.
This can lead to unexpected failures
if your branch is outdated, even when there are no explicit merge conflicts.

Update your branch only when needed: when you have merge conflicts, upstream CI is broken and blocking your green PR, or a maintainer requests it.
Avoid updating an already-green PR under review unless necessary.
During review, make incremental commits to address feedback.
Prefer to squash or rebase only at the end, or when a reviewer requests it.

When updating, use `git push --force-with-lease` and leave a brief comment explaining what changed.
Some repos prefer merging from `upstream/main` instead of rebasing;
follow the project's conventions.
See [keeping things up to date](git.md#keeping-things-up-to-date) for detailed instructions.

After rebasing, it's recommended to [run the relevant tests locally](tests/intro.md) to catch any issues before CI runs.

### r?

Your PR will be automatically assigned a reviewer.
You can override the reviewer using `r? @username`.
See [PR assignment](https://forge.rust-lang.org/triagebot/pr-assignment.html#usage) for details.

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

Please note that the reviewers are humans, who for the most part work on `rustc` in their free time.
This means that they can take some time to respond and review your PR.
It also means that reviewers can miss some PRs that are assigned to them.

To try to move PRs forward, the Triage WG regularly goes through all PRs that
are waiting for review and haven't been discussed for at least 2 weeks.
If you don't get a review within 2 weeks, feel free to ask the Triage WG on
Zulip ([#t-release/triage]).
They have knowledge of when to ping, who might be on vacation, etc.

The reviewer may request some changes using the GitHub code review interface.
They may also request special procedures for some PRs.
See [Crater] and [Breaking Changes] chapters for some examples of such procedures.

[r?]: https://github.com/rust-lang/rust/pull/78133#issuecomment-712692371
[#t-release/triage]: https://rust-lang.zulipchat.com/#narrow/stream/242269-t-release.2Ftriage
[Crater]: tests/crater.md

### CI

In addition to being reviewed by a human, pull requests are automatically tested,
thanks to continuous integration (CI).
Basically, every time you open and update
a pull request, CI builds the compiler and tests it against the
[compiler test suite], and also performs other tests such as checking that
your pull request is in compliance with Rust's style guidelines.

Running continuous integration tests allows PR authors to catch mistakes early
without going through a first review cycle, and also helps reviewers stay aware
of the status of a particular pull request.

Rust has plenty of CI capacity, and you should never have to worry about wasting
computational resources each time you push a change.
It is also perfectly fine
(and even encouraged!) to use the CI to test your changes if it can help your productivity.
In particular, we don't recommend running the full `./x test` suite locally,
since it takes a very long time to execute.
See the [Testing with CI] chapter for using Rust's CI to test your changes.

[Testing with CI]: tests/ci.md#testing-with-ci

### r+

After someone has reviewed your pull request, they will leave an annotation
on the pull request with an `r+`.
It will look something like this:

    @bors r+

This tells [@bors], our lovable integration bot, that your pull request has been approved.
The PR then enters the [merge queue], where [@bors]
will run *all* the tests on *every* platform we support.
If it all works out, [@bors] will merge your code into `main` and close the pull request.

Depending on the scale of the change, you may see a slightly different form of `r+`:

    @bors r+ rollup

The additional `rollup` tells [@bors] that this change should always be "rolled up".
Changes that are rolled up are tested and merged alongside other PRs, to speed the process up.
Typically, only small changes that are expected not to conflict
with one another are marked as "always roll up".

Be patient;
this can take a while and the queue can sometimes be long.
Also, note that PRs are never merged by hand.

[@rustbot]: https://github.com/rustbot
[@bors]: https://github.com/rust-lang/bors

### Opening a PR

You are now ready to file a pull request (PR)?
Great!
Here are a few points you should be aware of.

All pull requests should be filed against the `main` branch,
unless you know for sure that you should target a different branch.

Run some style checks before you submit the PR:

    ./x test tidy --bless

We recommend to make this check before every pull request (and every new commit in a pull request);
you can add [git hooks] before every push to make sure you never forget to make this check.
The CI will also run tidy and will fail if tidy fails.

Rust follows a _no merge-commit policy_,
meaning that when you encounter merge conflicts,
you are expected to always rebase instead of merging.
For example,
always use rebase when bringing the latest changes from the `main` branch to your feature branch.
If your PR contains merge commits, it will get marked as `has-merge-commits`.
Once you have removed the merge commits, e.g., through an interactive rebase, you
should remove the label again:

    @rustbot label -has-merge-commits

See [this chapter][labeling] for more details.

If you encounter merge conflicts or when a reviewer asks you to perform some
changes, your PR will get marked as `S-waiting-on-author`.
When you resolve them, you should use `@rustbot` to mark it as `S-waiting-on-review`:

    @rustbot ready

GitHub allows [closing issues using keywords][closing-keywords].
This feature should be used to keep the issue tracker tidy.
However, it is generally preferred
to put the "closes #123" text in the PR description rather than the commit message;
particularly during rebasing, citing the issue number in the commit can "spam"
the issue in question.

However, if your PR fixes a stable-to-beta or stable-to-stable regression and has
been accepted for a beta and/or stable backport (i.e., it is marked `beta-accepted`
and/or `stable-accepted`), please do *not* use any such keywords since we don't
want the corresponding issue to get auto-closed once the fix lands on `main`.
Please update the PR description while still mentioning the issue somewhere.
For example, you could write `Fixes (after beta backport) #NNN.`.

As for further actions, please keep a sharp look-out for a PR whose title begins with
`[beta]` or `[stable]` and which backports the PR in question.
When that one gets merged, the relevant issue can be closed.
The closing comment should mention all PRs that were involved.
If you don't have the permissions to close the issue, please
leave a comment on the original PR asking the reviewer to close it for you.

[labeling]: ./rustbot.md#issue-relabeling
[closing-keywords]: https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue

### Reverting a PR

See ["Reverts"](https://forge.rust-lang.org/compiler/reviews.html#reverts) on Forge.

If a PR is large enough that it's hard to revert, it's ok to simply disable the trigger for the
problematic code, as shown in [#128271][#128271].
For MIR optimizations, we can also use the `-Zunsound-mir-opt` option to gate the mir-opt, as shown
in [#132356][#132356].

[revert policy]: https://forge.rust-lang.org/compiler/reviews.html?highlight=revert#reverts
[#128271]: https://github.com/rust-lang/rust/pull/128271
[#132356]: https://github.com/rust-lang/rust/pull/132356

## External dependencies

This section has moved to ["Using External Repositories"](./external-repos.md).

## LLM policy

See [Forge][LLM policy].

[LLM policy]: https://forge.rust-lang.org/policies/llm-usage.html

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
[create an issue]: https://github.com/rust-lang/rust/issues/new/choose
