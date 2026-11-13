# `track-diagnostics`

This feature is perma-unstable and has no tracking issue.

------------------------

This flag prints the source code span in the compiler where a diagnostic was generated, respecting [`#[track_caller]`][track_caller]. Note that this may be different from the place it was emitted.
For full documentation, see [the rustc-dev-guide][dev-guide-track-diagnostics].

[track_caller]: https://doc.rust-lang.org/reference/attributes/codegen.html#the-track_caller-attribute
[dev-guide-track-diagnostics]: https://rustc-dev-guide.rust-lang.org/compiler-debugging.html#getting-the-error-creation-location
