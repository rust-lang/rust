# `macro-stats`

This feature is perma-unstable and has no tracking issue.

----

Some macros, especially procedural macros, can generate a surprising amount of
code, which can slow down compile times. This is hard to detect because the
generated code is normally invisible to the programmer.

This flag helps identify such cases. When enabled, the compiler measures the
effect on code size of all used macros and prints a table summarizing that
effect. For each distinct macro, it counts how many times it is used, and the
net effect on code size (in terms of lines of code, and bytes of code). The
code size evaluation uses the compiler's internal pretty-printing, and so will
be independent of the formatting in the original code.

Note that the net effect of a macro may be negative. E.g. the `cfg!` and
`#[test]` macros often strip out code.

If a macro is identified as causing a large increase in code size, it is worth
using `cargo expand` to inspect the post-expansion code, which includes the
code produced by all macros. It may be possible to optimize the macro to
produce smaller code, or it may be possible to avoid using it altogether.
