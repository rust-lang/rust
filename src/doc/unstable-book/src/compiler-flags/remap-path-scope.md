# `remap-path-scope`

The tracking issue for this feature is: [#111540](https://github.com/rust-lang/rust/issues/111540).

------------------------

When the `--remap-path-prefix` option is passed to rustc, source path prefixes in all output will be affected by default.
The `--remap-path-scope` argument can be used in conjunction with `--remap-path-prefix` to determine paths in which output context should be affected.
This flag accepts a comma-separated list of values and may be specified multiple times, in which case the scopes are aggregated together. The valid scopes are:

- `macro` - apply remappings to the expansion of `std::file!()` macro. This is where paths in embedded panic messages come from
- `diagnostics` - apply remappings to printed compiler diagnostics
- `debuginfo` - apply remappings to debug informations
- `object` - apply remappings to all paths in compiled executables or libraries, but not elsewhere. Currently an alias for `macro,debuginfo`.
- `all` - an alias for all of the above, also equivalent to supplying only `--remap-path-prefix` without `--remap-path-scope`.

## Example
```sh
# This would produce an absolute path to main.rs in build outputs of
# "./main.rs".
rustc --remap-path-prefix=$(PWD)=/remapped -Zremap-path-scope=object main.rs
```
