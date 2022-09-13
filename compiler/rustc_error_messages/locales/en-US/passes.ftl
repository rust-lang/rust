-passes_previously_accepted =
    this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

-passes_see_issue =
    see issue #{$issue} <https://github.com/rust-lang/rust/issues/{$issue}> for more information

passes_outer_crate_level_attr =
    crate-level attribute should be an inner attribute: add an exclamation mark: `#![foo]`

passes_inner_crate_level_attr =
    crate-level attribute should be in the root module

passes_ignored_attr_with_macro = `#[{$sym}]` is ignored on struct fields, match arms and macro defs
    .warn = {-passes_previously_accepted}
    .note = {-passes_see_issue(issue: "80564")}

passes_ignored_attr = `#[{$sym}]` is ignored on struct fields and match arms
    .warn = {-passes_previously_accepted}
    .note = {-passes_see_issue(issue: "80564")}

passes_inline_ignored_function_prototype = `#[inline]` is ignored on function prototypes

passes_inline_ignored_constants = `#[inline]` is ignored on constants
    .warn = {-passes_previously_accepted}
    .note = {-passes_see_issue(issue: "65833")}

passes_inline_not_fn_or_closure = attribute should be applied to function or closure
    .label = not a function or closure

passes_no_coverage_ignored_function_prototype = `#[no_coverage]` is ignored on function prototypes

passes_no_coverage_propagate =
    `#[no_coverage]` does not propagate into items and must be applied to the contained functions directly

passes_no_coverage_fn_defn = `#[no_coverage]` may only be applied to function definitions

passes_no_coverage_not_coverable = `#[no_coverage]` must be applied to coverable code
    .label = not coverable code

passes_should_be_applied_to_fn = attribute should be applied to a function definition
    .label = not a function definition

passes_naked_tracked_caller = cannot use `#[track_caller]` with `#[naked]`

passes_should_be_applied_to_struct_enum = attribute should be applied to a struct or enum
    .label = not a struct or enum

passes_should_be_applied_to_trait = attribute should be applied to a trait
    .label = not a trait

passes_target_feature_on_statement = {passes_should_be_applied_to_fn}
    .warn = {-passes_previously_accepted}
    .label = {passes_should_be_applied_to_fn.label}

passes_should_be_applied_to_static = attribute should be applied to a static
    .label = not a static

passes_doc_expect_str = doc {$attr_name} attribute expects a string: #[doc({$attr_name} = "a")]

passes_doc_alias_empty = {$attr_str} attribute cannot have empty value

passes_doc_alias_bad_char = {$char_} character isn't allowed in {$attr_str}

passes_doc_alias_start_end = {$attr_str} cannot start or end with ' '

passes_doc_alias_bad_location = {$attr_str} isn't allowed on {$location}

passes_doc_alias_not_an_alias = {$attr_str} is the same as the item's name

passes_doc_alias_duplicated = doc alias is duplicated
    .label = first defined here

passes_doc_alias_not_string_literal = `#[doc(alias("a"))]` expects string literals

passes_doc_alias_malformed =
    doc alias attribute expects a string `#[doc(alias = "a")]` or a list of strings `#[doc(alias("a", "b"))]`

passes_doc_keyword_empty_mod = `#[doc(keyword = "...")]` should be used on empty modules

passes_doc_keyword_not_mod = `#[doc(keyword = "...")]` should be used on modules

passes_doc_keyword_invalid_ident = `{$doc_keyword}` is not a valid identifier

passes_doc_fake_variadic_not_valid =
    `#[doc(fake_variadic)]` must be used on the first of a set of tuple or fn pointer trait impls with varying arity

passes_doc_keyword_only_impl = `#[doc(keyword = "...")]` should be used on impl blocks

passes_doc_inline_conflict_first = this attribute...
passes_doc_inline_conflict_second = ...conflicts with this attribute
passes_doc_inline_conflict = conflicting doc inlining attributes
    .help = remove one of the conflicting attributes

passes_doc_inline_only_use = this attribute can only be applied to a `use` item
    .label = only applicable on `use` items
    .not_a_use_item_label = not a `use` item
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#inline-and-no_inline> for more information

passes_doc_attr_not_crate_level =
    `#![doc({$attr_name} = "...")]` isn't allowed as a crate-level attribute

passes_attr_crate_level = this attribute can only be applied at the crate level
    .suggestion = to apply to the crate, use an inner attribute
    .help = to apply to the crate, use an inner attribute
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#at-the-crate-level> for more information

passes_doc_test_unknown = unknown `doc(test)` attribute `{$path}`

passes_doc_test_takes_list = `#[doc(test(...)]` takes a list of attributes

passes_doc_primitive = `doc(primitive)` should never have been stable

passes_doc_test_unknown_any = unknown `doc` attribute `{$path}`

passes_doc_test_unknown_spotlight = unknown `doc` attribute `{$path}`
    .note = `doc(spotlight)` was renamed to `doc(notable_trait)`
    .suggestion = use `notable_trait` instead
    .no_op_note = `doc(spotlight)` is now a no-op

passes_doc_test_unknown_include = unknown `doc` attribute `{$path}`
    .suggestion = use `doc = include_str!` instead

passes_doc_invalid = invalid `doc` attribute

passes_pass_by_value = `pass_by_value` attribute should be applied to a struct, enum or type alias
    .label = is not a struct, enum or type alias

passes_allow_incoherent_impl =
    `rustc_allow_incoherent_impl` attribute should be applied to impl items.
    .label = the only currently supported targets are inherent methods

passes_has_incoherent_inherent_impl =
    `rustc_has_incoherent_inherent_impls` attribute should be applied to types or traits.
    .label = only adts, extern types and traits are supported

passes_must_use_async =
    `must_use` attribute on `async` functions applies to the anonymous `Future` returned by the function, not the value within
    .label = this attribute does nothing, the `Future`s returned by async functions are already `must_use`

passes_must_use_no_effect = `#[must_use]` has no effect when applied to {$article} {$target}

passes_must_not_suspend = `must_not_suspend` attribute should be applied to a struct, enum, or trait
    .label = is not a struct, enum, or trait

passes_cold = {passes_should_be_applied_to_fn}
    .warn = {-passes_previously_accepted}
    .label = {passes_should_be_applied_to_fn.label}

passes_link = attribute should be applied to an `extern` block with non-Rust ABI
    .warn = {-passes_previously_accepted}
    .label = not an `extern` block

passes_link_name = attribute should be applied to a foreign function or static
    .warn = {-passes_previously_accepted}
    .label = not a foreign function or static
    .help = try `#[link(name = "{$value}")]` instead

passes_no_link = attribute should be applied to an `extern crate` item
    .label = not an `extern crate` item

passes_export_name = attribute should be applied to a free function, impl method or static
    .label = not a free function, impl method or static

passes_rustc_layout_scalar_valid_range_not_struct = attribute should be applied to a struct
    .label = not a struct

passes_rustc_layout_scalar_valid_range_arg = expected exactly one integer literal argument

passes_rustc_legacy_const_generics_only = #[rustc_legacy_const_generics] functions must only have const generics
    .label = non-const generic parameter

passes_rustc_legacy_const_generics_index = #[rustc_legacy_const_generics] must have one index for each generic parameter
    .label = generic parameters

passes_rustc_legacy_const_generics_index_exceed = index exceeds number of arguments
    .label = there {$arg_count ->
        [one] is
        *[other] are
    } only {$arg_count} {$arg_count ->
        [one] argument
        *[other] arguments
    }

passes_rustc_legacy_const_generics_index_negative = arguments should be non-negative integers

passes_rustc_dirty_clean = attribute requires -Z query-dep-graph to be enabled

passes_link_section = attribute should be applied to a function or static
    .warn = {-passes_previously_accepted}
    .label = not a function or static

passes_no_mangle_foreign = `#[no_mangle]` has no effect on a foreign {$foreign_item_kind}
    .warn = {-passes_previously_accepted}
    .label = foreign {$foreign_item_kind}
    .note = symbol names in extern blocks are not mangled
    .suggestion = remove this attribute

passes_no_mangle = attribute should be applied to a free function, impl method or static
    .warn = {-passes_previously_accepted}
    .label = not a free function, impl method or static

passes_repr_ident = meta item in `repr` must be an identifier

passes_repr_conflicting = conflicting representation hints

passes_used_static = attribute must be applied to a `static` variable

passes_used_compiler_linker = `used(compiler)` and `used(linker)` can't be used together

passes_allow_internal_unstable = attribute should be applied to a macro
    .label = not a macro

passes_debug_visualizer_placement = attribute should be applied to a module

passes_debug_visualizer_invalid = invalid argument
    .note_1 = expected: `natvis_file = "..."`
    .note_2 = OR
    .note_3 = expected: `gdb_script_file = "..."`

passes_rustc_allow_const_fn_unstable = attribute should be applied to `const fn`
    .label = not a `const fn`

passes_rustc_std_internal_symbol = attribute should be applied to functions or statics
    .label = not a function or static

passes_const_trait = attribute should be applied to a trait

passes_stability_promotable = attribute cannot be applied to an expression

passes_deprecated = attribute is ignored here

passes_macro_use = `#[{$name}]` only has an effect on `extern crate` and modules

passes_macro_export = `#[macro_export]` only has an effect on macro definitions

passes_plugin_registrar = `#[plugin_registrar]` only has an effect on functions

passes_unused_empty_lints_note = attribute `{$name}` with an empty list has no effect

passes_unused_no_lints_note = attribute `{$name}` without any lints has no effect

passes_unused_default_method_body_const_note =
    `default_method_body_is_const` has been replaced with `#[const_trait]` on traits

passes_unused = unused attribute
    .suggestion = remove this attribute

passes_non_exported_macro_invalid_attrs = attribute should be applied to function or closure
    .label = not a function or closure

passes_unused_duplicate = unused attribute
    .suggestion = remove this attribute
    .note = attribute also specified here
    .warn = {-passes_previously_accepted}

passes_unused_multiple = multiple `{$name}` attributes
    .suggestion = remove this attribute
    .note = attribute also specified here

passes_rustc_lint_opt_ty = `#[rustc_lint_opt_ty]` should be applied to a struct
    .label = not a struct

passes_rustc_lint_opt_deny_field_access = `#[rustc_lint_opt_deny_field_access]` should be applied to a field
    .label = not a field

passes_link_ordinal = attribute should be applied to a foreign function or static
    .label = not a foreign function or static

passes_collapse_debuginfo = `collapse_debuginfo` attribute should be applied to macro definitions
    .label = not a macro definition
