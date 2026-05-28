//@ compile-flags: -Cllvm-args=-not-a-real-llvm-arg
//@ normalize-stderr: "--help" -> "-help"
//@ normalize-stderr: "\n(\n|.)*" -> ""
//@ ignore-backends: gcc

// I'm seeing "--help" locally, but "-help" in CI, so I'm normalizing it to just "-help".

// Note that the rustc-supplied "program name", given when invoking LLVM, is used by LLVM to
// generate user-facing error messages and a usage (--help) messages. If the program name is
// `rustc`, the usage message in response to `--llvm-args="--help"` starts with:
// ```
//   USAGE: rustc [options]
// ```
// followed by the list of options not to `rustc` but to `llvm`.
//
// On the other hand, if the program name is set to `rustc -Cllvm-args="..." with`, the usage
// message is more clear:
// ```
//   USAGE: rustc -Cllvm-args="..." with [options]
// ```
// This test captures the effect of the current program name setting on LLVM command line
// error messages.
fn main() {}
