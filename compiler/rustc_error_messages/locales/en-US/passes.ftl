-passes-previously-accepted =
    this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

-passes-see-issue =
    see issue #{$issue} <https://github.com/rust-lang/rust/issues/{$issue}> for more information

passes-outer-crate-level-attr =
    crate-level attribute should be an inner attribute: add an exclamation mark: `#![foo]`

passes-inner-crate-level-attr =
    crate-level attribute should be in the root module

passes-ignored-attr-with-macro = `#[{$sym}]` is ignored on struct fields, match arms and macro defs
    .warn = {-passes-previously-accepted}
    .note = {-passes-see-issue(issue: "80564")}

passes-ignored-attr = `#[{$sym}]` is ignored on struct fields and match arms
    .warn = {-passes-previously-accepted}
    .note = {-passes-see-issue(issue: "80564")}

passes-inline-ignored-function-prototype = `#[inline]` is ignored on function prototypes

passes-inline-ignored-constants = `#[inline]` is ignored on constants
    .warn = {-passes-previously-accepted}
    .note = {-passes-see-issue(issue: "65833")}

passes-inline-not-fn-or-closure = attribute should be applied to function or closure
    .label = not a function or closure

passes-no-coverage-ignored-function-prototype = `#[no_coverage]` is ignored on function prototypes

passes-no-coverage-propagate =
    `#[no_coverage]` does not propagate into items and must be applied to the contained functions directly

passes-no-coverage-fn-defn = `#[no_coverage]` may only be applied to function definitions

passes-no-coverage-not-coverable = `#[no_coverage]` must be applied to coverable code
    .label = not coverable code

passes-should-be-applied-to-fn = attribute should be applied to a function definition
    .label = not a function definition

passes-naked-tracked-caller = cannot use `#[track_caller]` with `#[naked]`

passes-should-be-applied-to-struct-enum = attribute should be applied to a struct or enum
    .label = not a struct or enum

passes-should-be-applied-to-trait = attribute should be applied to a trait
    .label = not a trait

passes-target-feature-on-statement = {passes-should-be-applied-to-fn}
    .warn = {-passes-previously-accepted}
    .label = {passes-should-be-applied-to-fn.label}

passes-should-be-applied-to-static = attribute should be applied to a static
    .label = not a static

passes-doc-expect-str = doc {$attr_name} attribute expects a string: #[doc({$attr_name} = "a")]

passes-doc-alias-empty = {$attr_str} attribute cannot have empty value

passes-doc-alias-bad-char = {$char_} character isn't allowed in {$attr_str}

passes-doc-alias-start-end = {$attr_str} cannot start or end with ' '

passes-doc-alias-bad-location = {$attr_str} isn't allowed on {$location}

passes-doc-alias-not-an-alias = {$attr_str} is the same as the item's name

passes-doc-alias-duplicated = doc alias is duplicated
    .label = first defined here

passes-doc-alias-not-string-literal = `#[doc(alias("a"))]` expects string literals

passes-doc-alias-malformed =
    doc alias attribute expects a string `#[doc(alias = "a")]` or a list of strings `#[doc(alias("a", "b"))]`

passes-doc-keyword-empty-mod = `#[doc(keyword = "...")]` should be used on empty modules

passes-doc-keyword-not-mod = `#[doc(keyword = "...")]` should be used on modules

passes-doc-keyword-invalid-ident = `{$doc_keyword}` is not a valid identifier

passes-doc-fake-variadic-not-valid =
    `#[doc(fake_variadic)]` must be used on the first of a set of tuple or fn pointer trait impls with varying arity

passes-doc-keyword-only-impl = `#[doc(keyword = "...")]` should be used on impl blocks

passes-doc-inline-conflict-first = this attribute...
passes-doc-inline-conflict-second = ...conflicts with this attribute
passes-doc-inline-conflict = conflicting doc inlining attributes
    .help = remove one of the conflicting attributes

passes-doc-inline-only-use = this attribute can only be applied to a `use` item
    .label = only applicable on `use` items
    .not-a-use-item-label = not a `use` item
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#inline-and-no_inline> for more information

passes-doc-attr-not-crate-level =
    `#![doc({$attr_name} = "...")]` isn't allowed as a crate-level attribute

passes-attr-crate-level = this attribute can only be applied at the crate level
    .suggestion = to apply to the crate, use an inner attribute
    .help = to apply to the crate, use an inner attribute
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#at-the-crate-level> for more information

passes-doc-test-unknown = unknown `doc(test)` attribute `{$path}`

passes-doc-test-takes-list = `#[doc(test(...)]` takes a list of attributes

passes-doc-primitive = `doc(primitive)` should never have been stable

passes-doc-test-unknown-any = unknown `doc` attribute `{$path}`

passes-doc-test-unknown-spotlight = unknown `doc` attribute `{$path}`
    .note = `doc(spotlight)` was renamed to `doc(notable_trait)`
    .suggestion = use `notable_trait` instead
    .no-op-note = `doc(spotlight)` is now a no-op

passes-doc-test-unknown-include = unknown `doc` attribute `{$path}`
    .suggestion = use `doc = include_str!` instead

passes-doc-invalid = invalid `doc` attribute

passes-pass-by-value = `pass_by_value` attribute should be applied to a struct, enum or type alias
    .label = is not a struct, enum or type alias

passes-allow-incoherent-impl =
    `rustc_allow_incoherent_impl` attribute should be applied to impl items.
    .label = the only currently supported targets are inherent methods

passes-has-incoherent-inherent-impl =
    `rustc_has_incoherent_inherent_impls` attribute should be applied to types or traits.
    .label = only adts, extern types and traits are supported

passes-must-use-async =
    `must_use` attribute on `async` functions applies to the anonymous `Future` returned by the function, not the value within
    .label = this attribute does nothing, the `Future`s returned by async functions are already `must_use`

passes-must-use-no-effect = `#[must_use]` has no effect when applied to {$article} {$target}

passes-must-not-suspend = `must_not_suspend` attribute should be applied to a struct, enum, or trait
    .label = is not a struct, enum, or trait

passes-cold = {passes-should-be-applied-to-fn}
    .warn = {-passes-previously-accepted}
    .label = {passes-should-be-applied-to-fn.label}

passes-link = attribute should be applied to an `extern` block with non-Rust ABI
    .warn = {-passes-previously-accepted}
    .label = not an `extern` block

passes-link-name = attribute should be applied to a foreign function or static
    .warn = {-passes-previously-accepted}
    .label = not a foreign function or static
    .help = try `#[link(name = "{$value}")]` instead

passes-no-link = attribute should be applied to an `extern crate` item
    .label = not an `extern crate` item

passes-export-name = attribute should be applied to a free function, impl method or static
    .label = not a free function, impl method or static

passes-rustc-layout-scalar-valid-range-not-struct = attribute should be applied to a struct
    .label = not a struct

passes-rustc-layout-scalar-valid-range-arg = expected exactly one integer literal argument

passes-rustc-legacy-const-generics-only = #[rustc_legacy_const_generics] functions must only have const generics
    .label = non-const generic parameter

passes-rustc-legacy-const-generics-index = #[rustc_legacy_const_generics] must have one index for each generic parameter
    .label = generic parameters

passes-rustc-legacy-const-generics-index-exceed = index exceeds number of arguments
    .label = there {$arg_count ->
        [one] is
        *[other] are
    } only {$arg_count} {$arg_count ->
        [one] argument
        *[other] arguments
    }

passes-rustc-legacy-const-generics-index-negative = arguments should be non-negative integers

passes-rustc-dirty-clean = attribute requires -Z query-dep-graph to be enabled

passes-link-section = attribute should be applied to a function or static
    .warn = {-passes-previously-accepted}
    .label = not a function or static

passes-no-mangle-foreign = `#[no_mangle]` has no effect on a foreign {$foreign_item_kind}
    .warn = {-passes-previously-accepted}
    .label = foreign {$foreign_item_kind}
    .note = symbol names in extern blocks are not mangled
    .suggestion = remove this attribute

passes-no-mangle = attribute should be applied to a free function, impl method or static
    .warn = {-passes-previously-accepted}
    .label = not a free function, impl method or static

passes-repr-ident = meta item in `repr` must be an identifier

passes-repr-conflicting = conflicting representation hints

passes-used-static = attribute must be applied to a `static` variable

passes-used-compiler-linker = `used(compiler)` and `used(linker)` can't be used together

passes-allow-internal-unstable = attribute should be applied to a macro
    .label = not a macro

passes-debug-visualizer-placement = attribute should be applied to a module

passes-debug-visualizer-invalid = invalid argument
    .note-1 = expected: `natvis_file = "..."`
    .note-2 = OR
    .note-3 = expected: `gdb_script_file = "..."`

passes-rustc-allow-const-fn-unstable = attribute should be applied to `const fn`
    .label = not a `const fn`

passes-rustc-std-internal-symbol = attribute should be applied to functions or statics
    .label = not a function or static

passes-const-trait = attribute should be applied to a trait

passes-stability-promotable = attribute cannot be applied to an expression

passes-deprecated = attribute is ignored here

passes-macro-use = `#[{$name}]` only has an effect on `extern crate` and modules

passes-macro-export = `#[macro_export]` only has an effect on macro definitions

passes-plugin-registrar = `#[plugin_registrar]` only has an effect on functions

passes-unused-empty-lints-note = attribute `{$name}` with an empty list has no effect

passes-unused-no-lints-note = attribute `{$name}` without any lints has no effect

passes-unused-default-method-body-const-note =
    `default_method_body_is_const` has been replaced with `#[const_trait]` on traits

passes-unused = unused attribute
    .suggestion = remove this attribute

passes-non-exported-macro-invalid-attrs = attribute should be applied to function or closure
    .label = not a function or closure

passes-unused-duplicate = unused attribute
    .suggestion = remove this attribute
    .note = attribute also specified here
    .warn = {-passes-previously-accepted}

passes-unused-multiple = multiple `{$name}` attributes
    .suggestion = remove this attribute
    .note = attribute also specified here

passes-rustc-lint-opt-ty = `#[rustc_lint_opt_ty]` should be applied to a struct
    .label = not a struct

passes-rustc-lint-opt-deny-field-access = `#[rustc_lint_opt_deny_field_access]` should be applied to a field
    .label = not a field
