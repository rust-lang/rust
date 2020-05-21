# `strip`

The tracking issue for this feature is: [#72110](https://github.com/rust-lang/rust/issues/72110).

------------------------

Option `-Z strip=val` controls stripping of debuginfo and similar auxiliary data from binaries
during linking.

Supported values for this option are:

- `none` - debuginfo and symbols (if they exist) are copied to the produced binary or separate files
depending on the target (e.g. `.pdb` files in case of MSVC).
- `debuginfo` - debuginfo sections and debuginfo symbols from the symbol table section
are stripped at link time and are not copied to the produced binary or separate files.
- `symbols` - same as `debuginfo`, but the rest of the symbol table section is stripped as well
if the linker supports it.
