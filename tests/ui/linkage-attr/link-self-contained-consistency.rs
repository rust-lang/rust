// Checks that self-contained linking components cannot be both enabled and disabled at the same
// time on the CLI.

// check-fail
// revisions: one many
// [one] compile-flags: -Clink-self-contained=-linker -Clink-self-contained=+linker -Zunstable-options
// [many] compile-flags: -Clink-self-contained=+linker,+crto -Clink-self-contained=-linker,-crto -Zunstable-options
// ignore-tidy-linelength

fn main() {}
