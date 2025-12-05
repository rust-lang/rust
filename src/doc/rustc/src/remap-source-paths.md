# Remap source paths

`rustc` supports remapping source paths prefixes **as a best effort** in all compiler generated
output, including compiler diagnostics, debugging information, macro expansions, etc.

This is useful for normalizing build products, for example by removing the current directory
out of the paths emitted into object files.

The remapping is done via the `--remap-path-prefix` option.

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
