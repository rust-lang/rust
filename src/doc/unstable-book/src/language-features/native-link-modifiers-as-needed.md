# `native_link_modifiers_as_needed`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers_as_needed` feature allows you to use the `as-needed` modifier.

`as-needed` is only compatible with the `dynamic` and `framework` linking kinds. Using any other kind will result in a compiler error.

`+as-needed` means that the library will be actually linked only if it satisfies some undefined symbols at the point at which it is specified on the command line, making it similar to static libraries in this regard.

This modifier translates to `--as-needed` for ld-like linkers, and to `-dead_strip_dylibs` / `-needed_library` / `-needed_framework` for ld64.
The modifier does nothing for linkers that don't support it (e.g. `link.exe`).

The default for this modifier is unclear, some targets currently specify it as `+as-needed`, some do not. We may want to try making `+as-needed` a default for all targets.
