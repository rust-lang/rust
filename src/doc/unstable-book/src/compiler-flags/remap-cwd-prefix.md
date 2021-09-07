# `remap-cwd-prefix`

The tracking issue for this feature is: [#87325](https://github.com/rust-lang/rust/issues/87325).

------------------------

This flag will rewrite absolute paths under the current working directory,
replacing the current working directory prefix with a specified value.

The given value may be absolute or relative, or empty. This switch takes
precidence over `--remap-path-prefix` in case they would both match a given
path.

This flag helps to produce deterministic output, by removing the current working
directory from build output, while allowing the command line to be universally
reproducible, such that the same execution will work on all machines, regardless
of build environment.

## Example
```sh
# This would produce an absolute path to main.rs in build outputs of
# "./main.rs".
rustc -Z remap-cwd-prefix=. main.rs
```
