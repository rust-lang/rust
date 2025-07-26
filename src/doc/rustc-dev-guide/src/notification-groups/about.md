# Notification groups

The **notification groups** are an easy way to help out with rustc in a
"piece-meal" fashion, without committing to a larger project.
Notification groups are **[easy to join](#join)** (just submit a PR!)
and joining does not entail any particular commitment.

Once you [join a notification group](#join), you will be added to
a list that receives pings on github whenever a new issue is found
that fits the notification group's criteria. If you are interested, you
can then [claim the issue] and start working on it.

Of course, you don't have to wait for new issues to be tagged! If you
prefer, you can use the Github label for a notification group to
search for existing issues that haven't been claimed yet.

[claim the issue]: https://forge.rust-lang.org/triagebot/issue-assignment.html

## List of notification groups

Here's the list of the notification groups:
- [Apple](./apple.md)
- [ARM](./arm.md)
- [Emscripten](./emscripten.md)
- [RISC-V](./risc-v.md)
- [WASI](./wasi.md)
- [WebAssembly](./wasm.md)
- [Windows](./windows.md)
- [Rust for Linux](./rust-for-linux.md)

## What issues are a good fit for notification groups?

Notification groups tend to get pinged on **isolated** bugs,
particularly those of **middle priority**:

- By **isolated**, we mean that we do not expect large-scale refactoring
  to be required to fix the bug.
- By **middle priority**, we mean that we'd like to see the bug fixed,
  but it's not such a burning problem that we are dropping everything
  else to fix it. The danger with such bugs, of course, is that they
  can accumulate over time, and the role of the notification group is
  to try and stop that from happening!

<a id="join"></a>

## Joining a notification group

To join a notification group, you just have to open a PR adding your
Github username to the appropriate file in the Rust team repository.
See the "example PRs" below to get a precise idea and to identify the
file to edit.

Also, if you are not already a member of a Rust team then -- in addition
to adding your name to the file -- you have to checkout the repository and
run the following command:

```bash
cargo run add-person $your_user_name
```

Example PRs:

* [Example of adding yourself to the Apple group.](https://github.com/rust-lang/team/pull/1434)
* [Example of adding yourself to the ARM group.](https://github.com/rust-lang/team/pull/358)
* [Example of adding yourself to the Emscripten group.](https://github.com/rust-lang/team/pull/1579)
* [Example of adding yourself to the RISC-V group.](https://github.com/rust-lang/team/pull/394)
* [Example of adding yourself to the WASI group.](https://github.com/rust-lang/team/pull/1580)
* [Example of adding yourself to the WebAssembly group.](https://github.com/rust-lang/team/pull/1581)
* [Example of adding yourself to the Windows group.](https://github.com/rust-lang/team/pull/348)

## Tagging an issue for a notification group

To tag an issue as appropriate for a notification group, you give
[rustbot] a [`ping`] command with the name of the notification
group. For example:

```text
@rustbot ping apple
@rustbot ping arm
@rustbot ping emscripten
@rustbot ping risc-v
@rustbot ping wasi
@rustbot ping wasm
@rustbot ping windows
```

To make some commands shorter and easier to remember, there are aliases,
defined in the [`triagebot.toml`] file. For example, all of these commands
are equivalent and will ping the Apple group:

```text
@rustbot ping apple
@rustbot ping macos
@rustbot ping ios
```

Keep in mind that these aliases are meant to make humans' life easier.
They might be subject to change. If you need to ensure that a command
will always be valid, prefer the full invocations over the aliases.

**Note though that this should only be done by compiler team members
or contributors, and is typically done as part of compiler team
triage.**

[rustbot]: https://github.com/rust-lang/triagebot/
[`ping`]: https://forge.rust-lang.org/triagebot/pinging.html
[`triagebot.toml`]: https://github.com/rust-lang/rust/blob/master/triagebot.toml
