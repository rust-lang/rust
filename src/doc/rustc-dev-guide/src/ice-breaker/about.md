# ICE-breakers

The **ICE-breaker groups** are an easy way to help out with rustc in a
"piece-meal" fashion, without committing to a larger project.
ICE-breaker groups are **[easy to join](#join)** (just submit a PR!)
and joining does not entail any particular commitment.

Once you [join an ICE ICE-breaker group](#join), you will be added to
a list that receives pings on github whenever a new issue is found
that fits the ICE-breaker group's criteria. If you are interested, you
can then [claim the issue] and start working on it.

Of course, you don't have to wait for new issues to be tagged! If you
prefer, you can use the Github label for an ICE-breaker group to
search for existing issues that haven't been claimed yet.

[claim the issue]: https://github.com/rust-lang/triagebot/wiki/Assignment

## What issues are a good fit for ICE-breaker groups?

"ICE-breaker issues" are intended to be **isolated** bugs of **middle
priority**:

- By **isolated**, we mean that we do not expect large-scale refactoring
  to be required to fix the bug.
- By **middle priority**, we mean that we'd like to see the bug fixed,
  but it's not such a burning problem that we are dropping everything
  else to fix it. The danger with such bugs, of course, is that they
  can accumulate over time, and the role of the ICE-breaker groups is
  to try and stop that from happening!

<a name="join"></a>

## Joining an ICE-breaker group

To join an ICE-breaker group, you just have to open a PR adding your
Github username to the appropriate file in the Rust team  repository.
See the "example PRs" below to get a precise idea and to identify the
file to edit.

Also, if you are not already a member of a Rust team then -- in addition
to adding your name to the file -- you have to checkout the repository and
run the following command:

```bash
cargo run add-person $your_user_name
```

Example PRs:

* [Example of adding yourself to the LLVM ICE-breakers.](https://github.com/rust-lang/team/pull/140)

## Tagging an issue for an ICE-breaker group

To tag an issue as appropriate for an ICE-breaker group, you give
[rustbot] a [`ping`] command with the name of the ICE-breakers
team. For example:

```text
@rustbot ping icebreakers-llvm
```

**Note though that this should only be done by compiler team members
or contributors, and is typically done as part of compiler team
triage.**

[rustbot]: https://github.com/rust-lang/triagebot/
[`ping`]: https://github.com/rust-lang/triagebot/wiki/Pinging
