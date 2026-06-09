# `move_size_limit`

--------------------

The `-Zmove-size-limit=N` compiler flag enables `large_assignments` lints which
will warn when moving objects whose size exceeds `N` bytes.

Lint warns only about moves in functions that participate in code generation.
Consequently it will be ineffective for compiler invocation that emit
metadata only, i.e., `cargo check` like workflows.
