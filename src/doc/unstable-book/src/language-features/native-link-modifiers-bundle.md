# `native_link_modifiers_bundle`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers_bundle` feature allows you to use the `bundle` modifier.

Only compatible with the `static` linking kind. Using any other kind will result in a compiler error.

`+bundle` means objects from the static library are bundled into the produced crate (a rlib, for example) and are used from this crate later during linking of the final binary.

`-bundle` means the static library is included into the produced rlib "by name" and object files from it are included only during linking of the final binary, the file search by that name is also performed during final linking.

This modifier is supposed to supersede the `static-nobundle` linking kind defined by [RFC 1717](https://github.com/rust-lang/rfcs/pull/1717).

The default for this modifier is currently `+bundle`, but it could be changed later on some future edition boundary.
