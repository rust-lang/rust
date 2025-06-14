expand_arg_not_attributes =
    second argument must be `attributes`

expand_attr_no_arguments =
    attribute must have either one or two arguments

expand_attribute_meta_item =
    attribute must be a meta item, not a literal

expand_attribute_single_word =
    attribute must only be a single word

expand_attributes_on_expressions_experimental =
    attributes on expressions are experimental
    .help_outer_doc = `///` is used for outer documentation comments; for a plain comment, use `//`
    .help_inner_doc = `//!` is used for inner documentation comments; for a plain comment, use `//` by removing the `!` or inserting a space in between them: `// !`

expand_attributes_wrong_form =
    attribute must be of form: `attributes(foo, bar)`

expand_cannot_be_name_of_macro =
    `{$trait_ident}` cannot be a name of {$macro_type} macro

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
    .note = removed in {$removed_rustc_version} (you are using {$current_rustc_version}){$pull_note}
    .reason = {$reason}

expand_glob_delegation_outside_impls =
    glob delegation is only supported in impls

expand_glob_delegation_traitless_qpath =
    qualified path without a trait in glob delegation

expand_helper_attribute_name_invalid =
    `{$name}` cannot be a name of derive helper attribute

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

expand_macro_body_stability =
    macros cannot have body stability attributes
    .label = invalid body stability attribute
    .label2 = body stability attribute affects this macro

expand_macro_const_stability =
    macros cannot have const stability attributes
    .label = invalid const stability attribute
    .label2 = const stability attribute affects this macro

expand_macro_expands_to_match_arm = macros cannot expand to match arms

expand_malformed_feature_attribute =
    malformed `feature` attribute input
    .expected = expected just one word

expand_meta_var_dif_seq_matchers = {$msg}

expand_meta_var_expr_unrecognized_var =
    variable `{$key}` is not recognized in meta-variable expression

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
    cannot declare a non-inline module inside a block unless it has a path attribute
    .note = maybe `use` the module `{$name}` instead of redeclaring it

expand_module_multiple_candidates =
    file for module `{$name}` found at both "{$default_path}" and "{$secondary_path}"
    .help = delete or rename one of them to remove the ambiguity

expand_must_repeat_once =
    this must repeat at least once

expand_non_inline_modules_in_proc_macro_input_are_unstable =
    non-inline modules in proc macro input are unstable

expand_not_a_meta_item =
    not a meta item

expand_only_one_word =
    must only be one word

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

expand_unsupported_key_value =
    key-value macro attributes are not supported

expand_var_still_repeating =
    variable `{$ident}` is still repeating at this depth

expand_wrong_fragment_kind =
    non-{$kind} macro in {$kind} position: {$name}
