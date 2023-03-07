expand_explain_doc_comment_outer =
    outer doc comments expand to `#[doc = "..."]`, which is what this macro attempted to match

expand_explain_doc_comment_inner =
    inner doc comments expand to `#![doc = "..."]`, which is what this macro attempted to match

expand_expr_repeat_no_syntax_vars =
    attempted to repeat an expression containing no syntax variables matched as repeating at this depth

expand_must_repeat_once =
    this must repeat at least once

expand_count_repetition_misplaced =
    `count` can not be placed inside the inner-most repetition

expand_meta_var_expr_unrecognized_var =
    variable `{$key}` is not recognized in meta-variable expression

expand_var_still_repeating =
    variable '{$ident}' is still repeating at this depth

expand_meta_var_dif_seq_matchers = {$msg}

expand_macro_const_stability =
    macros cannot have const stability attributes
    .label = invalid const stability attribute
    .label2 = const stability attribute affects this macro

expand_macro_body_stability =
    macros cannot have body stability attributes
    .label = invalid body stability attribute
    .label2 = body stability attribute affects this macro

expand_resolve_relative_path =
    cannot resolve relative path in non-file source `{$path}`

expand_attr_no_arguments =
    attribute must have either one or two arguments

expand_not_a_meta_item =
    not a meta item

expand_only_one_word =
    must only be one word

expand_cannot_be_name_of_macro =
    `{$trait_ident}` cannot be a name of {$macro_type} macro

expand_arg_not_attributes =
    second argument must be `attributes`

expand_attributes_wrong_form =
    attribute must be of form: `attributes(foo, bar)`

expand_attribute_meta_item =
    attribute must be a meta item, not a literal

expand_attribute_single_word =
    attribute must only be a single word

expand_helper_attribute_name_invalid =
    `{$name}` cannot be a name of derive helper attribute

expand_expected_comma_in_list =
    expected token: `,`

expand_only_one_argument =
    {$name} takes 1 argument

expand_takes_no_arguments =
    {$name} takes no arguments

expand_feature_included_in_edition =
    the feature `{$feature}` is included in the Rust {$edition} edition

expand_feature_removed =
    feature has been removed
    .label = feature has been removed
    .reason = {$reason}

expand_feature_not_allowed =
    the feature `{$name}` is not in the list of allowed features

expand_recursion_limit_reached =
    recursion limit reached while expanding `{$descr}`
    .help = consider increasing the recursion limit by adding a `#![recursion_limit = "{$suggested_limit}"]` attribute to your crate (`{$crate_name}`)

expand_malformed_feature_attribute =
    malformed `feature` attribute input
    .expected = expected just one word

expand_remove_expr_not_supported =
    removing an expression is not supported in this position

expand_invalid_cfg_no_parens = `cfg` is not followed by parentheses
expand_invalid_cfg_no_predicate = `cfg` predicate is not specified
expand_invalid_cfg_multiple_predicates = multiple `cfg` predicates are specified
expand_invalid_cfg_predicate_literal = `cfg` predicate key cannot be a literal
expand_invalid_cfg_expected_syntax = expected syntax is

expand_wrong_fragment_kind =
    non-{$kind} macro in {$kind} position: {$name}

expand_unsupported_key_value =
    key-value macro attributes are not supported

expand_incomplete_parse =
    macro expansion ignores token `{$token}` and any following
    .label = caused by the macro expansion here
    .note = the usage of `{$macro_path}!` is likely invalid in {$kind_name} context
    .suggestion_add_semi = you might be missing a semicolon here

expand_remove_node_not_supported =
    removing {$descr} is not supported in this position

expand_module_circular =
    circular modules: {$modules}

expand_module_in_block =
    cannot declare a non-inline module inside a block unless it has a path attribute
    .note = maybe `use` the module `{$name}` instead of redeclaring it

expand_module_file_not_found =
    file not found for module `{$name}`
    .help = to create the module `{$name}`, create file "{$default_path}" or "{$secondary_path}"

expand_module_multiple_candidates =
    file for module `{$name}` found at both "{$default_path}" and "{$secondary_path}"
    .help = delete or rename one of them to remove the ambiguity

expand_trace_macro = trace_macro

expand_proc_macro_panicked =
    proc macro panicked
    .help = message: {$message}

expand_proc_macro_derive_tokens =
    proc-macro derive produced unparseable tokens
