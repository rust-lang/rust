# Symbol Mangling

[Symbol name mangling] is used by `rustc` to encode a unique name for symbols that are used during code generation.
The encoded names are used by the linker to associate the name with the thing it refers to.

The method for mangling the names can be controlled with the [`-C symbol-mangling-version`] option.

[Symbol name mangling]: https://en.wikipedia.org/wiki/Name_mangling
[`-C symbol-mangling-version`]: ../codegen-options/index.md#symbol-mangling-version

## Per-item control

The [`#[no_mangle]` attribute][reference-no_mangle] can be used on items to disable name mangling on that item.

The [`#[export_name]`attribute][reference-export_name] can be used to specify the exact name that will be used for a function or static.

Items listed in an [`extern` block][reference-extern-block] use the identifier of the item without mangling to refer to the item.
The [`#[link_name]` attribute][reference-link_name] can be used to change that name.

<!--
FIXME: This is incomplete for wasm, per https://github.com/rust-lang/rust/blob/d4c364347ce65cf083d4419195b8232440928d4d/compiler/rustc_symbol_mangling/src/lib.rs#L191-L210
-->

[reference-no_mangle]: ../../reference/abi.html#the-no_mangle-attribute
[reference-export_name]: ../../reference/abi.html#the-export_name-attribute
[reference-link_name]: ../../reference/items/external-blocks.html#the-link_name-attribute
[reference-extern-block]: ../../reference/items/external-blocks.html

## Decoding

The encoded names may need to be decoded in some situations.
For example, debuggers and other tooling may need to demangle the name so that it is more readable to the user.
Recent versions of `gdb` and `lldb` have built-in support for demangling Rust identifiers.
In situations where you need to do your own demangling, the [`rustc-demangle`] crate can be used to programmatically demangle names.
[`rustfilt`] is a CLI tool which can demangle names.

An example of running rustfilt:

```text
$ rustfilt _RNvCskwGfYPst2Cb_3foo16example_function
foo::example_function
```

[`rustc-demangle`]: https://crates.io/crates/rustc-demangle
[`rustfilt`]: https://crates.io/crates/rustfilt

## Mangling versions

`rustc` supports different mangling versions which encode the names in different ways.
The legacy version (which is currently the default) is not described here.
The "v0" mangling scheme addresses several limitations of the legacy format,
and is described in the [v0 Symbol Format](v0.md) chapter.
