# LLVM Icebreakers Notification group

**Github Label:** [A-LLVM] <br>
**Ping command:** `@rustbot ping icebreakers-llvm`

[A-LLVM]: https://github.com/rust-lang/rust/labels/A-LLVM

*Note*: this notification group is *not* the same as the LLVM working group
(WG-llvm).

The "LLVM Icebreakers Notification Group" are focused on bugs that center around
LLVM. These bugs often arise because of LLVM optimizations gone awry, or as the
result of an LLVM upgrade. The goal here is:

- to determine whether the bug is a result of us generating invalid LLVM IR,
  or LLVM misoptimizing;
- if the former, to fix our IR;
- if the latter, to try and file a bug on LLVM (or identify an existing bug).

The group may also be asked to weigh in on other sorts of LLVM-focused
questions.

## Helpful tips and options

The ["Debugging LLVM"][d] section of the
rustc-dev-guide gives a step-by-step process for how to help debug bugs
caused by LLVM. In particular, it discusses how to emit LLVM IR, run
the LLVM IR optimization pipelines, and so forth. You may also find
it useful to look at the various codegen options listed under `-C help`
and the internal options under `-Z help` -- there are a number that
pertain to LLVM (just search for LLVM).

[d]: ../backend/debugging.md

## If you do narrow to an LLVM bug

The ["Debugging LLVM"][d] section also describes what to do once
you've identified the bug.
