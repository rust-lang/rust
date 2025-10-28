# Remap source paths

`rustc` supports remapping source paths prefixes **as a best effort** in all compiler generated
output, including compiler diagnostics, debugging information, macro expansions, etc.

This is useful for normalizing build products, for example by removing the current directory
out of the paths emitted into object files.

The remapping is done via the `--remap-path-prefix` flag and can be customized via the `--remap-path-scope` flag.

## `--remap-path-prefix`

It takes a value of the form `FROM=TO` where a path prefix equal to `FROM` is rewritten
to the value `TO`. `FROM` may itself contain an `=` symbol, but `TO` value may not.

The replacement is purely textual, with no consideration of the current system's path separator.

When multiple remappings are given and several of them match, the **last** matching one is applied.

### Example

```bash
rustc --remap-path-prefix "/home/user/project=/redacted"
```

This example replaces all occurrences of `/home/user/project` in emitted paths with `/redacted`.

## `--remap-path-scope`

Defines which scopes of paths should be remapped by `--remap-path-prefix`.

This flag accepts a comma-separated list of values and may be specified multiple times, in which case the scopes are aggregated together.

The valid scopes are:

- `macro` - apply remappings to the expansion of `std::file!()` macro. This is where paths in embedded panic messages come from
- `diagnostics` - apply remappings to printed compiler diagnostics
- `debuginfo` - apply remappings to debug information
- `coverage` - apply remappings to coverage information
- `object` - apply remappings to all paths in compiled executables or libraries, but not elsewhere. Currently an alias for `macro,coverage,debuginfo`.
- `all` (default) - an alias for all of the above, also equivalent to supplying only `--remap-path-prefix` without `--remap-path-scope`.

The scopes accepted by `--remap-path-scope` are not exhaustive - new scopes may be added in future releases for eventual stabilisation.
This implies that the `all` scope can correspond to different scopes between releases.

### Example

```sh
# With `object` scope only the build outputs will be remapped, the diagnostics won't be remapped.
rustc --remap-path-prefix=$(PWD)=/remapped --remap-path-scope=object main.rs
```

## Caveats and Limitations

### Linkers generated paths

On some platforms like `x86_64-pc-windows-msvc`, the linker may embed absolute host paths and compiler
arguments into debug info files (like `.pdb`) independently of `rustc`.

Additionally, on Apple platforms, linkers generate [OSO entries] which are not remapped by the compiler
and need to be manually remapped with `-oso_prefix`.

The `--remap-path-prefix` option does not affect these linker-generated paths.

### Textual replacement only

The remapping is strictly textual and does not account for different path separator conventions across
platforms. Care must be taken when specifying prefixes, especially on Windows where both `/` and `\` may
appear in paths.

### External tools

Paths introduced by external tools or environment variables may not be covered by `--remap-path-prefix`
unless explicitly accounted for.

For example, generated code introduced by Cargo's build script may still contain un-remapped paths.

[OSO entries]: https://wiki.dwarfstd.org/Apple%27s_%22Lazy%22_DWARF_Scheme.md
