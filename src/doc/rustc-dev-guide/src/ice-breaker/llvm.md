# LLVM ICE-breakers

**Github Label:** [ICEBreaker-LLVM]

[ICEBreaker-LLVM]: https://github.com/rust-lang/rust/labels/ICEBreaker-LLVM

The "LLVM ICE-breakers" are focused on bugs that center around LLVM.
These bugs often arise because of LLVM optimizations gone awry, or as
the result of an LLVM upgrade. The goal here is:

- to determine whether the bug is a result of us generating invalid LLVM IR,
  or LLVM misoptimizing;
- if the former, to fix our IR;
- if the latter, to try and file a bug on LLVM (or identify an existing bug).

## Helpful tips and options

The ["Debugging LLVM"][d] section of the
rustc-guide gives a step-by-step process for how to help debug bugs
caused by LLVM. In particular, it discusses how to emit LLVM IR, run
the LLVM IR optimization pipeliness, and so forth. You may also find
it useful to look at the various codegen options listed under `-Chelp`
and the internal options under `-Zhelp` -- there are a number that
pertain to LLVM (just search for LLVM).

[d]: ../codegen/debugging.md

## If you do narrow to an LLVM bug

The ["Debugging LLVM"][d] section also describes what to do once
you've identified the bug.
