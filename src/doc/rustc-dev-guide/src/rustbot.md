# Mastering @rustbot

`@rustbot` (also known as `triagebot`) is a utility robot that is mostly used to
allow any contributor to achieve certain tasks that would normally require GitHub
membership to the `rust-lang` organization. Its most interesting features for
contributors to `rustc` are issue claiming and relabeling.

## Issue claiming

`@rustbot` exposes a command that allows anyone to assign an issue to themselves.
If you see an issue you want to work on, you can send the following message as a
comment on the issue at hand:

    @rustbot claim

This will tell `@rustbot` to assign the issue to you if it has no assignee yet.
Note that because of some GitHub restrictions, you may be assigned indirectly,
i.e. `@rustbot` will assign itself as a placeholder and edit the top comment to
reflect the fact that the issue is now assigned to you.

If you want to unassign from an issue, `@rustbot` has a different command:

    @rustbot release-assignment

## Issue relabeling

Changing labels for an issue or PR is also normally reserved for members of the
organization. However, `@rustbot` allows you to relabel an issue yourself, only
with a few restrictions. This is mostly useful in two cases:

**Helping with issue triage**: Rust's issue tracker has more than 5,000 open
issues at the time of this writing, so labels are the most powerful tool that we
have to keep it as tidy as possible. You don't need to spend hours in the issue tracker
to triage issues, but if you open an issue, you should feel free to label it if
you are comfortable with doing it yourself.

**Updating the status of a PR**: We use "status labels" to reflect the status of
PRs. For example, if your PR has merge conflicts, it will automatically be assigned
the `S-waiting-on-author`, and reviewers might not review it until you rebase your
PR. Once you do rebase your branch, you should change the labels yourself to remove
the `S-waiting-on-author` label and add back `S-waiting-on-review`. In this case,
the `@rustbot` command will look like this:

    @rustbot label -S-waiting-on-author +S-waiting-on-review

The syntax for this command is pretty loose, so there are other variants of this
command invocation. There are also some shortcuts to update labels,
for instance `@rustbot ready` will do the same thing with above command.
For more details, see [the docs page about labeling][labeling] and [shortcuts][shortcuts].

[labeling]: https://forge.rust-lang.org/triagebot/labeling.html
[shortcuts]: https://forge.rust-lang.org/triagebot/shortcuts.html

## Other commands

If you are interested in seeing what `@rustbot` is capable of, check out its [documentation],
which is meant as a reference for the bot and should be kept up to date every time the
bot gets an upgrade.

`@rustbot` is maintained by the Release team. If you have any feedback regarding
existing commands or suggestions for new commands, feel free to reach out
[on Zulip][zulip] or file an issue in [the triagebot repository][repo]

[documentation]: https://forge.rust-lang.org/triagebot/index.html
[zulip]: https://rust-lang.zulipchat.com/#narrow/stream/224082-t-release.2Ftriagebot
[repo]: https://github.com/rust-lang/triagebot/
