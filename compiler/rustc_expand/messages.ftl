expand_attributes_on_expressions_experimental =
    attributes on expressions are experimental
    .help_outer_doc = `///` is used for outer documentation comments; for a plain comment, use `//`
    .help_inner_doc = `//!` is used for inner documentation comments; for a plain comment, use `//` by removing the `!` or inserting a space in between them: `// !`

expand_cfg_attr_no_attributes = `#[cfg_attr]` does not expand to any attributes

expand_collapse_debuginfo_illegal =
    illegal value for attribute #[collapse_debuginfo(no|external|yes)]

expand_count_repetition_misplaced =
    `count` can not be placed inside the innermost repetition

expand_crate_name_in_cfg_attr =
    `crate_name` within an `#![cfg_attr]` attribute is forbidden

expand_crate_type_in_cfg_attr =
    `crate_type` within an `#![cfg_attr]` attribute is forbidden

expand_custom_attribute_panicked =
    custom attribute panicked
    .help = message: {$message}

expand_duplicate_matcher_binding = duplicate matcher binding
    .label = duplicate binding
    .label2 = previous binding

expand_empty_delegation_mac =
    empty {$kind} delegation is not supported

expand_expected_paren_or_brace =
    expected `(` or `{"{"}`, found `{$token}`

expand_explain_doc_comment_inner =
    inner doc comments expand to `#![doc = "..."]`, which is what this macro attempted to match

expand_explain_doc_comment_outer =
    outer doc comments expand to `#[doc = "..."]`, which is what this macro attempted to match

expand_expr_repeat_no_syntax_vars =
    attempted to repeat an expression containing no syntax variables matched as repeating at this depth

expand_feature_not_allowed =
    the feature `{$name}` is not in the list of allowed features

expand_feature_removed =
    feature has been removed
    .label = feature has been removed
    .note = removed in {$removed_rustc_version}{$pull_note}
    .reason = {$reason}

expand_glob_delegation_outside_impls =
    glob delegation is only supported in impls

expand_glob_delegation_traitless_qpath =
    qualified path without a trait in glob delegation

expand_incomplete_parse =
    macro expansion ignores {$descr} and any tokens following
    .label = caused by the macro expansion here
    .note = the usage of `{$macro_path}!` is likely invalid in {$kind_name} context
    .suggestion_add_semi = you might be missing a semicolon here

expand_invalid_cfg_expected_syntax = expected syntax is

expand_invalid_cfg_multiple_predicates = multiple `cfg` predicates are specified
expand_invalid_cfg_no_parens = `cfg` is not followed by parentheses
expand_invalid_cfg_no_predicate = `cfg` predicate is not specified
expand_invalid_cfg_predicate_literal = `cfg` predicate key cannot be a literal

expand_invalid_fragment_specifier =
    invalid fragment specifier `{$fragment}`
    .help = {$help}

expand_macro_args_bad_delim = `{$rule_kw}` rule argument matchers require parentheses
expand_macro_args_bad_delim_sugg = the delimiters should be `(` and `)`

expand_macro_body_stability =
    macros cannot have body stability attributes
    .label = invalid body stability attribute
    .label2 = body stability attribute affects this macro

expand_macro_call_unused_doc_comment = unused doc comment
    .label = rustdoc does not generate documentation for macro invocations
    .help = to document an item produced by a macro, the macro must produce the documentation as part of its expansion

expand_macro_const_stability =
    macros cannot have const stability attributes
    .label = invalid const stability attribute
    .label2 = const stability attribute affects this macro

expand_macro_expands_to_match_arm = macros cannot expand to match arms

expand_malformed_feature_attribute =
    malformed `feature` attribute input
    .expected = expected just one word

expand_meta_var_dif_seq_matchers = {$msg}

expand_metavar_still_repeating = variable `{$ident}` is still repeating at this depth
    .label = expected repetition

expand_metavariable_wrong_operator = meta-variable repeats with different Kleene operator
    .binder_label = expected repetition
    .occurrence_label = conflicting repetition

expand_missing_fragment_specifier = missing fragment specifier
    .note = fragment specifiers must be provided
    .suggestion_add_fragspec = try adding a specifier here
    .valid = {$valid}

expand_module_circular =
    circular modules: {$modules}

expand_module_file_not_found =
    file not found for module `{$name}`
    .help = to create the module `{$name}`, create file "{$default_path}" or "{$secondary_path}"
    .note = if there is a `mod {$name}` elsewhere in the crate already, import it with `use crate::...` instead

expand_module_in_block =
    cannot declare a file module inside a block unless it has a path attribute
    .help = maybe `use` the module `{$name}` instead of redeclaring it
    .note = file modules are usually placed outside of blocks, at the top level of the file

expand_module_multiple_candidates =
    file for module `{$name}` found at both "{$default_path}" and "{$secondary_path}"
    .help = delete or rename one of them to remove the ambiguity

expand_must_repeat_once =
    this must repeat at least once

expand_mve_extra_tokens =
    unexpected trailing tokens
    .label = for this metavariable expression
    .range = the `{$name}` metavariable expression takes between {$min_or_exact_args} and {$max_args} arguments
    .exact = the `{$name}` metavariable expression takes {$min_or_exact_args ->
        [zero] no arguments
        [one] a single argument
        *[other] {$min_or_exact_args} arguments
    }
    .suggestion = try removing {$extra_count ->
        [one] this token
        *[other] these tokens
    }

expand_mve_missing_paren =
    expected `(`
    .label = for this this metavariable expression
    .unexpected = unexpected token
    .note = metavariable expressions use function-like parentheses syntax
    .suggestion = try adding parentheses

expand_mve_unrecognized_expr =
    unrecognized metavariable expression
    .label = not a valid metavariable expression
    .note = valid metavariable expressions are {$valid_expr_list}

expand_mve_unrecognized_var =
    variable `{$key}` is not recognized in meta-variable expression

expand_non_inline_modules_in_proc_macro_input_are_unstable =
    non-inline modules in proc macro input are unstable

expand_or_patterns_back_compat = the meaning of the `pat` fragment specifier is changing in Rust 2021, which may affect this macro
    .suggestion = use pat_param to preserve semantics

expand_proc_macro_back_compat = using an old version of `{$crate_name}`
    .note = older versions of the `{$crate_name}` crate no longer compile; please update to `{$crate_name}` v{$fixed_version}, or switch to one of the `{$crate_name}` alternatives

expand_proc_macro_derive_panicked =
    proc-macro derive panicked
    .help = message: {$message}

expand_proc_macro_derive_tokens =
    proc-macro derive produced unparsable tokens

expand_proc_macro_panicked =
    proc macro panicked
    .help = message: {$message}

expand_recursion_limit_reached =
    recursion limit reached while expanding `{$descr}`
    .help = consider increasing the recursion limit by adding a `#![recursion_limit = "{$suggested_limit}"]` attribute to your crate (`{$crate_name}`)

expand_remove_expr_not_supported =
    removing an expression is not supported in this position

expand_remove_node_not_supported =
    removing {$descr} is not supported in this position

expand_resolve_relative_path =
    cannot resolve relative path in non-file source `{$path}`

expand_trace_macro = trace_macro

expand_trailing_semi_macro = trailing semicolon in macro used in expression position
    .note1 = macro invocations at the end of a block are treated as expressions
    .note2 = to ignore the value produced by the macro, add a semicolon after the invocation of `{$name}`

expand_unknown_macro_variable = unknown macro variable `{$name}`

expand_unsupported_key_value =
    key-value macro attributes are not supported

expand_unused_builtin_attribute = unused attribute `{$attr_name}`
    .note = the built-in attribute `{$attr_name}` will be ignored, since it's applied to the macro invocation `{$macro_name}`
    .suggestion = remove the attribute

expand_wrong_fragment_kind =
    non-{$kind} macro in {$kind} position: {$name}
