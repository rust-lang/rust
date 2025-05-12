# Attributes

Attributes come in two types: *inert* (or *built-in*) and *active* (*non-builtin*).

## Builtin/inert attributes

These attributes are defined in the compiler itself, in
[`compiler/rustc_feature/src/builtin_attrs.rs`][builtin_attrs].

Examples include `#[allow]` and `#[macro_use]`.

[builtin_attrs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_feature/builtin_attrs/index.html

These attributes have several important characteristics:
* They are always in scope, and do not participate in typical path-based resolution.
* They cannot be renamed. For example, `use allow as foo` will compile, but writing `#[foo]` will
  produce an error.
* They are 'inert', meaning they are left as-is by the macro expansion code.
  As a result, any behavior comes as a result of the compiler explicitly checking for their presence.
  For example, lint-related code explicitly checks for `#[allow]`, `#[warn]`, `#[deny]`, and
  `#[forbid]`, rather than the behavior coming from the expansion of the attributes themselves.

## 'Non-builtin'/'active' attributes

These attributes are defined by a crate - either the standard library, or a proc-macro crate.

**Important**: Many non-builtin attributes, such as `#[derive]`, are still considered part of the
core Rust language. However, they are **not** called 'builtin attributes', since they have a
corresponding definition in the standard library.

Definitions of non-builtin attributes take two forms:

1. Proc-macro attributes, defined via a function annotated with `#[proc_macro_attribute]` in a
   proc-macro crate.
2. AST-based attributes, defined in the standard library. These attributes have special 'stub'
   macros defined in places like [`library/core/src/macros/mod.rs`][core_macros].

[core_macros]:  https://github.com/rust-lang/rust/blob/master/library/core/src/macros/mod.rs

These definitions exist to allow the macros to participate in typical path-based resolution - they
can be imported, re-exported, and renamed just like any other item definition. However, the body of
the definition is empty. Instead, the macro is annotated with the `#[rustc_builtin_macro]`
attribute, which tells the compiler to run a corresponding function in `rustc_builtin_macros`.

All non-builtin attributes have the following characteristics:
* Like all other definitions (e.g. structs), they must be brought into scope via an import.
  Many standard library attributes are included in the prelude - this is why writing `#[derive]`
  works without an import.
* They participate in macro expansion. The implementation of the macro may leave the attribute
  target unchanged, modify the target, produce new AST nodes, or remove the target entirely.
