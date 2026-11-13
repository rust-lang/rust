# Attributes

Attributes come in two types: *inert* (or *built-in*) and *active* (*non-builtin*).

## Builtin/inert attributes

These attributes are defined in the compiler itself, in
[`compiler/rustc_feature/src/builtin_attrs.rs`][builtin_attrs] and in the [attribute parsers][attr_parsing].

Examples include `#[allow]` and `#[macro_use]`.

[builtin_attrs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_feature/builtin_attrs/index.html
[attr_parsing]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_attr_parsing/index.html

These attributes have several important characteristics:
* They are always in scope, and do not participate in typical path-based resolution.
* They cannot be renamed.
  For example, `use allow as foo` will compile, but writing `#[foo]` will produce an error.
* They are 'inert', meaning they are left as-is by the macro expansion code.
  As a result, any behavior comes as a result of the compiler explicitly checking for their presence.
  For example, lint-related code explicitly checks for `#[allow]`, `#[warn]`, `#[deny]`, and
  `#[forbid]`, rather than the behavior coming from the expansion of the attributes themselves.

For more information on these attributes, see the chapter about [attribute parsing][attr-parsing-chapter].

[attr-parsing-chapter]: ./hir/attribute-parsing.md

### Note on adding new builtin attributes

<div class="warning">

**Warning: Name resolution ambiguity potential when adding new builtin attributes**

Please note that adding **new builtin attributes** (whose name is not reserved, i.e. a new builtin
attribute whose name does not start with `rustc`), even if *unstable*-gated, can introduce breakage
from name resolution ambiguity in stable code if (1) the stable code has a macro of the same name
which gets re-exported, or (2) or a proc-macro derive helper attribute of the same name.

Typically, the builtin attributes probably has to start out as `#[rustc_foo]` instead of `#[foo]` to
avoid colliding with user-defined macros and proc-macro helper attributes.
Then, prior to stabilization,
a rename to `#[foo]` should be done separately with a crater run to assess fallout,
with a deliberate breakage FCP proposal for T-lang to consider.

Remember also that crater is *not* exhaustive and does not contain all existing stable code.

See:
- [Built-in attributes are treated differently vs prelude attributes, unstable built-in attributes
  can name-collide with stable macro, and built-in attributes can break back-compat
  #134963](https://github.com/rust-lang/rust/issues/134963) and backlinks within this issue,
  including design discussions on how to fix this kind of breakage hazard.
- [Broken build after updating: coverage is ambiguous; ambiguous because of a name conflict with a
  builtin attribute #121157](https://github.com/rust-lang/rust/issues/121157).
- [Regression: align is ambiguous #143834](https://github.com/rust-lang/rust/issues/143834).
</div>

## 'Non-builtin'/'active' attributes

These attributes are defined by a crate - either the standard library, or a proc-macro crate.

**Important**: Many non-builtin attributes, such as `#[derive]`, are still considered part of the
core Rust language.
However, they are **not** called 'builtin attributes', since they have a
corresponding definition in the standard library.

Definitions of non-builtin attributes take two forms:

1. Proc-macro attributes, defined via a function annotated with `#[proc_macro_attribute]` in a
   proc-macro crate.
2. AST-based attributes, defined in the standard library.
   These attributes have special 'stub'
   macros defined in places like [`library/core/src/macros/mod.rs`][core_macros].

[core_macros]:  https://github.com/rust-lang/rust/blob/HEAD/library/core/src/macros/mod.rs

These definitions exist to allow the macros to participate in typical path-based resolution - they
can be imported, re-exported, and renamed just like any other item definition.
However, the body of the definition is empty.
Instead, the macro is annotated with the `#[rustc_builtin_macro]`
attribute, which tells the compiler to run a corresponding function in `rustc_builtin_macros`.

All non-builtin attributes have the following characteristics:
* Like all other definitions (e.g. structs), they must be brought into scope via an import.
  Many standard library attributes are included in the prelude - this is why writing `#[derive]`
  works without an import.
* They participate in macro expansion.
  The implementation of the macro may leave the attribute
  target unchanged, modify the target, produce new AST nodes, or remove the target entirely.
