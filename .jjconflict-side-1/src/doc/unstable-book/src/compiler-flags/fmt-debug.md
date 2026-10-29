# `fmt-debug`

The tracking issue for this feature is: [#129709](https://github.com/rust-lang/rust/issues/129709).

------------------------

Option `-Z fmt-debug=val` controls verbosity of derived `Debug` implementations
and debug formatting in format strings (`{:?}`).

* `full` — `#[derive(Debug)]` prints types recursively. This is the default behavior.

* `shallow` — `#[derive(Debug)]` prints only the type name, or name of a variant of a fieldless enums. Details of the `Debug` implementation are not stable and may change in the future. Behavior of custom `fmt::Debug` implementations is not affected.

* `none` — `#[derive(Debug)]` does not print anything at all. `{:?}` in formatting strings has no effect.
  This option may reduce size of binaries, and remove occurrences of type names in the binary that are not removed by striping symbols. However, it may also cause `panic!` and `assert!` messages to be incomplete.
