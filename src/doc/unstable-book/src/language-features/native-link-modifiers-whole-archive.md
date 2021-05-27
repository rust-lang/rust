# `native_link_modifiers_whole_archive`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers_whole_archive` feature allows you to use the `whole-archive` modifier.

Only compatible with the `static` linking kind. Using any other kind will result in a compiler error.

`+whole-archive` means that the static library is linked as a whole archive without throwing any object files away.

This modifier translates to `--whole-archive` for `ld`-like linkers, to `/WHOLEARCHIVE` for `link.exe`, and to `-force_load` for `ld64`.
The modifier does nothing for linkers that don't support it.

The default for this modifier is `-whole-archive`.
