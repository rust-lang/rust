# `native_link_modifiers_verbatim`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers_verbatim` feature allows you to use the `verbatim` modifier.

`+verbatim` means that rustc itself won't add any target-specified library prefixes or suffixes (like `lib` or `.a`) to the library name, and will try its best to ask for the same thing from the linker.

For `ld`-like linkers rustc will use the `-l:filename` syntax (note the colon) when passing the library, so the linker won't add any prefixes or suffixes as well.
See [`-l namespec`](https://sourceware.org/binutils/docs/ld/Options.html) in ld documentation for more details.
For linkers not supporting any verbatim modifiers (e.g. `link.exe` or `ld64`) the library name will be passed as is.

The default for this modifier is `-verbatim`.

This RFC changes the behavior of `raw-dylib` linking kind specified by [RFC 2627](https://github.com/rust-lang/rfcs/pull/2627). The `.dll` suffix (or other target-specified suffixes for other targets) is now added automatically.
If your DLL doesn't have the `.dll` suffix, it can be specified with `+verbatim`.
