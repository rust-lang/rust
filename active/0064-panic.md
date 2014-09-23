- Start Date: 2014-09-23
- RFC PR #: [rust-lang/rfcs#221](https://github.com/rust-lang/rfcs/pull/221)
- Rust Issue #: [rust-lang/rust#17489](https://github.com/rust-lang/rust/issues/17489)

# Summary

Rename "task failure" to "task panic", and `fail!` to `panic!`.

# Motivation

The current terminology of "task failure" often causes problems when
writing or speaking about code. You often want to talk about the
possibility of an operation that returns a `Result` "failing", but
cannot because of the ambiguity with task failure. Instead, you have
to speak of "the failing case" or "when the operation does not
succeed" or other circumlocutions.

Likewise, we use a "Failure" header in rustdoc to describe when
operations may fail the task, but it would often be helpful to
separate out a section describing the "Err-producing" case.

We have been steadily moving away from task failure and toward
`Result` as an error-handling mechanism, so we should optimize our
terminology accordingly: `Result`-producing functions should be easy
to describe.

# Detailed design

Not much more to say here than is in the summary: rename "task
failure" to "task panic" in documentation, and `fail!` to `panic!` in
code.

The choice of `panic` emerged from a
[discuss thread](http://discuss.rust-lang.org/t/renaming-task-failure/310/20)
and
[workweek discussion](https://github.com/rust-lang/meeting-minutes/blob/master/workweek-2014-08-18/error-handling.md).
It has precedent in a language setting in Go, and of course goes back
to Kernel panics.

With this choice, we can use "failure" to refer to an operation that
produces `Err` or `None`, "panic" for unwinding at the task level, and
"abort" for aborting the entire process.

The connotations of panic seem fairly accurate: the process is not
immediately ending, but it is rapidly fleeing from some problematic
circumstance (by killing off tasks) until a recovery point.

# Drawbacks

The term "panic" is a bit informal, which some consider a drawback.

Making this change is likely to be a lot of work.

# Alternatives

Other choices include:

- `throw!` or `unwind!`. These options reasonably describe the current
  behavior of task failure, but "throw" suggests general exception
  handling, and both place the emphasis on the mechanism rather than
  the policy. We also are considering eventually adding a flag that
  allows `fail!` to abort the process, which would make these terms misleading.

- `abort!`. Ambiguous with process abort.

- `die!`. A reasonable choice, but it's not immediately obvious what
  is being killed.
