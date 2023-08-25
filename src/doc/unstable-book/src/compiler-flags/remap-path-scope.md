# `remap-path-scope`

The tracking issue for this feature is: [#111540](https://github.com/rust-lang/rust/issues/111540).

------------------------

When the `--remap-path-prefix` option is passed to rustc, source path prefixes in all output will be affected by default.
The `--remap-path-scope` argument can be used in conjunction with `--remap-path-prefix` to determine paths in which output context should be affected.
This flag accepts a comma-separated list of values and may be specified multiple times, in which case the scopes are aggregated together. The valid scopes are:

- `macro` - apply remappings to the expansion of `std::file!()` macro. This is where paths in embedded panic messages come from
- `diagnostics` - apply remappings to printed compiler diagnostics
- `unsplit-debuginfo` - apply remappings to debug information only when they are written to compiled executables or libraries, but not when they are in split debuginfo files
- `split-debuginfo` - apply remappings to debug information only when they are written to split debug information files, but not in compiled executables or libraries
- `split-debuginfo-path` - apply remappings to the paths pointing to split debug information files. Does nothing when these files are not generated.
- `object` - an alias for `macro,unsplit-debuginfo,split-debuginfo-path`. This ensures all paths in compiled executables or libraries are remapped, but not elsewhere.
- `all` and `true` - an alias for all of the above, also equivalent to supplying only `--remap-path-prefix` without `--remap-path-scope`.

## Example
```sh
# This would produce an absolute path to main.rs in build outputs of
# "./main.rs".
rustc --remap-path-prefix=$(PWD)=/remapped -Zremap-path-prefix=object main.rs
```
