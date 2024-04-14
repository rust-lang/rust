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

expand_concat_generated_invalid_ident = `{"${concat(..)}"}` is not generating a valid identifier

expand_concat_raw_ident = `{"${concat(..)}"}` currently does not support raw identifiers

expand_concat_too_few_args = `concat` must have at least two elements

expand_count_repetition_misplaced =
    `count` can not be placed inside the inner-most repetition

expand_count_with_comma_no_index = `count` followed by a comma must have an associated index indicating its depth

expand_custom_attribute_cannot_be_applied =
    custom attributes cannot be applied to {$kind ->
        [statement] statements
        *[expression] expressions
    }

expand_custom_attribute_panicked =
    custom attribute panicked
    .help = message: {$message}

expand_doc_comments_ignored_in_matcher_position = doc comments are ignored in matcher position

expand_dollar_or_metavar_in_lhs = unexpected token: {$token}
    .note = `$$` and meta-variable expressions are not allowed inside macro parameter definitions

expand_duplicate_binding_name = duplicated bind name: {$bind}

expand_duplicate_matcher_binding = duplicate matcher binding
    .label = duplicate binding
    .label2 = previous binding

expand_empty_delegation_mac =
    empty {$kind} delegation is not supported

expand_expected_comma = expected comma

expand_expected_identifier = expected identifier, found `{$found}`

expand_expected_paren_or_brace =
    expected `(` or `{"{"}`, found `{$token}`

expand_expected_repetition_operator = expected one of: `*`, `+`, or `?`

expand_explain_doc_comment_inner =
    inner doc comments expand to `#![doc = "..."]`, which is what this macro attempted to match

expand_explain_doc_comment_outer =
    outer doc comments expand to `#[doc = "..."]`, which is what this macro attempted to match

expand_expr_2021_unstable = fragment specifier `expr_2021` is unstable

expand_expr_repeat_no_syntax_vars =
    attempted to repeat an expression containing no syntax variables matched as repeating at this depth

expand_feature_not_allowed =
    the feature `{$name}` is not in the list of allowed features

expand_feature_removed =
    feature has been removed
    .label = feature has been removed
    .reason = {$reason}

expand_glob_delegation_outside_impls =
    glob delegation is only supported in impls

expand_glob_delegation_traitless_qpath =
    qualified path without a trait in glob delegation

expand_helper_attribute_name_invalid =
    `{$name}` cannot be a name of derive helper attribute

expand_incomplete_parse =
    macro expansion ignores token `{$token}` and any following
    .label = caused by the macro expansion here
    .note = the usage of `{$macro_path}!` is likely invalid in {$kind_name} context
    .suggestion_add_semi = you might be missing a semicolon here

expand_invalid_cfg_expected_syntax = expected syntax is

expand_invalid_cfg_multiple_predicates = multiple `cfg` predicates are specified
expand_invalid_cfg_no_parens = `cfg` is not followed by parentheses
expand_invalid_cfg_no_predicate = `cfg` predicate is not specified
expand_invalid_cfg_predicate_literal = `cfg` predicate key cannot be a literal

expand_invalid_concat_arg_type = metavariables of {"`${concat(..)}`"} must be of type `ident`, `literal` or `tt`
    .note = currently only string literals are supported

expand_invalid_follow = `${$name}:{$kind}` {$only_option ->
            [true] is
            *[false] may be
        } followed by `{$next}`, which is not allowed for `{$kind}` fragments
    .label = not allowed after `{$kind}` fragments
    .suggestion = try a `pat_param` fragment specifier instead
    .note = {$num_possible ->
            [one] only {$possible} is allowed after `{$kind}` fragments
            *[other] allowed there are: {$possible}
        }

expand_invalid_fragment_specifier =
    invalid fragment specifier `{$fragment}`
    .help_expr_2021 = fragment specifier `expr_2021` requires Rust 2021 or later

expand_label_conflicting_repetition = conflicting repetition

expand_label_error_while_parsing_argument = while parsing argument for this `{$kind}` macro fragment

expand_label_expected_repetition = expected repetition
expand_label_previous_declaration = previous declaration

expand_macro_body_stability =
    macros cannot have body stability attributes
    .label = invalid body stability attribute
    .label2 = body stability attribute affects this macro

expand_macro_const_stability =
    macros cannot have const stability attributes
    .label = invalid const stability attribute
    .label2 = const stability attribute affects this macro

expand_macro_expands_to_match_arm = macros cannot expand to match arms

expand_macro_rhs_must_be_delimited = macro rhs must be delimited

expand_malformed_feature_attribute =
    malformed `feature` attribute input
    .expected = expected just one word

expand_match_failure_missing_tokens = missing tokens in macro arguments
expand_match_failure_unexpected_token = no rules expected this token in macro call

expand_meta_var_dif_seq_matchers =
    meta-variable `{$var1_id}` repeats {$var1_len} {$var1_len ->
        [one] time
        *[count] times
    }, but `{$var2_id}` repeats {$var2_len} {$var2_len ->
        [one] time
        *[count] times
    }

expand_meta_var_expr_concat_unstable = the `concat` meta-variable expression is unstable

expand_meta_var_expr_depth_not_literal = meta-variable expression depth must be a literal

expand_meta_var_expr_depth_suffixed = only unsuffixed integer literals are supported in meta-variable expressions

expand_meta_var_expr_expected_identifier = expected identifier, found `{$found}`
    .suggestion = try removing `{$found}`

expand_meta_var_expr_needs_parens = meta-variable expression parameter must be wrapped in parentheses

expand_meta_var_expr_out_of_bounds = {$max ->
        [0] meta-variable expression `{$ty}` with depth parameter must be called inside of a macro repetition
        *[other] depth parameter of meta-variable expression `{$ty}` must be less than {$max}
    }

expand_meta_var_expr_unexpected_token = unexpected token: {$tt}
    .note = meta-variable expression must not have trailing tokens

expand_meta_var_expr_unrecognized_var =
    variable `{$key}` is not recognized in meta-variable expression

expand_meta_var_expr_unstable = meta-variable expressions are unstable

expand_missing_fragment_specifier = missing fragment specifier
    .note = fragment specifiers must be specified in the 2024 edition
    .suggestion_add_fragspec = try adding a specifier here

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

expand_multiple_parsing_options =
    local ambiguity when calling macro `{$macro_name}`: multiple parsing options: built-in NTs {$nts}{$n ->
        [0] .
        [one] {""} or {$n} other option
        *[other] {""} or {$n} other options
    }

expand_multiple_successful_parses = ambiguity: multiple successful parses

expand_multiple_transparency_attrs = multiple macro transparency attributes

expand_must_repeat_once =
    this must repeat at least once

expand_nested_meta_var_expr_without_dollar = meta-variables within meta-variable expressions must be referenced using a dollar sign

expand_non_inline_modules_in_proc_macro_input_are_unstable =
    non-inline modules in proc macro input are unstable

expand_not_a_meta_item =
    not a meta item

expand_only_one_word =
    must only be one word

expand_parse_failure_expected_token = expected `{$expected}`, found `{$found}`
expand_parse_failure_unexpected_eof = unexpected end of macro invocation
expand_parse_failure_unexpected_token = no rules expected the token `{$found}`

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

expand_question_mark_with_separator = the `?` macro repetition operator does not take a separator

expand_recursion_limit_reached =
    recursion limit reached while expanding `{$descr}`
    .help = consider increasing the recursion limit by adding a `#![recursion_limit = "{$suggested_limit}"]` attribute to your crate (`{$crate_name}`)

expand_remove_expr_not_supported =
    removing an expression is not supported in this position

expand_remove_node_not_supported =
    removing {$descr} is not supported in this position

expand_repetition_matches_empty_token_tree = repetition matches empty token tree

expand_resolve_relative_path =
    cannot resolve relative path in non-file source `{$path}`

expand_trace_macro = trace_macro

expand_unbalanced_delims_around_matcher = invalid macro matcher; matchers must be contained in balanced delimiters

expand_unknown_macro_transparency = unknown macro transparency: `{$value}`

expand_unrecognized_meta_var_expr = unrecognized meta-variable expression
    .help = supported expressions are count, ignore, index and len

expand_unsupported_concat_elem = expected identifier or string literal

expand_unsupported_key_value =
    key-value macro attributes are not supported

expand_valid_fragment_names_2021 = valid fragment specifiers are `ident`, `block`, `stmt`, `expr`, `expr_2021`, `pat`, `ty`, `lifetime`, `literal`, `path`, `meta`, `tt`, `item` and `vis`
expand_valid_fragment_names_other = valid fragment specifiers are `ident`, `block`, `stmt`, `expr`, `pat`, `ty`, `lifetime`, `literal`, `path`, `meta`, `tt`, `item` and `vis`

expand_var_still_repeating =
    variable `{$ident}` is still repeating at this depth

expand_wrong_fragment_kind =
    non-{$kind} macro in {$kind} position: {$name}
