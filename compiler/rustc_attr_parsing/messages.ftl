attr_parsing_as_needed_compatibility =
    linking modifier `as-needed` is only compatible with `dylib`, `framework` and `raw-dylib` linking kinds

attr_parsing_bundle_needs_static =
    linking modifier `bundle` is only compatible with `static` linking kind

attr_parsing_cfg_predicate_identifier =
    `cfg` predicate key must be an identifier

attr_parsing_deprecated_item_suggestion =
    suggestions on deprecated items are unstable
    .help = add `#![feature(deprecated_suggestion)]` to the crate root
    .note = see #94785 for more details

attr_parsing_empty_attribute =
    unused attribute
    .suggestion = {$valid_without_list ->
        [true] remove these parentheses
        *[other] remove this attribute
    }
    .note = {$valid_without_list ->
        [true] using `{$attr_path}` with an empty list is equivalent to not using a list at all
        *[other] using `{$attr_path}` with an empty list has no effect
    }


attr_parsing_empty_confusables =
    expected at least one confusable name
attr_parsing_empty_link_name =
    link name must not be empty
    .label = empty link name

attr_parsing_expected_one_cfg_pattern =
    expected 1 cfg-pattern

attr_parsing_expected_single_version_literal =
    expected single version literal

attr_parsing_expected_version_literal =
    expected a version literal

attr_parsing_expects_feature_list =
    `{$name}` expects a list of feature names

attr_parsing_expects_features =
    `{$name}` expects feature names

attr_parsing_ill_formed_attribute_input = {$num_suggestions ->
        [1] attribute must be of the form {$suggestions}
        *[other] valid forms for the attribute are {$suggestions}
    }

attr_parsing_import_name_type_raw =
    import name type can only be used with link kind `raw-dylib`

attr_parsing_import_name_type_x86 =
    import name type is only supported on x86

attr_parsing_incompatible_wasm_link =
    `wasm_import_module` is incompatible with other arguments in `#[link]` attributes

attr_parsing_incorrect_repr_format_align_one_arg =
    incorrect `repr(align)` attribute format: `align` takes exactly one argument in parentheses

attr_parsing_incorrect_repr_format_expect_literal_integer =
    incorrect `repr(align)` attribute format: `align` expects a literal integer as argument

attr_parsing_incorrect_repr_format_generic =
    incorrect `repr({$repr_arg})` attribute format
    .suggestion = use parentheses instead

attr_parsing_incorrect_repr_format_packed_expect_integer =
    incorrect `repr(packed)` attribute format: `packed` expects a literal integer as argument

attr_parsing_incorrect_repr_format_packed_one_or_zero_arg =
    incorrect `repr(packed)` attribute format: `packed` takes exactly one parenthesized argument, or no parentheses at all

attr_parsing_invalid_alignment_value =
    invalid alignment value: {$error_part}

attr_parsing_invalid_attr_unsafe = `{$name}` is not an unsafe attribute
    .label = this is not an unsafe attribute
    .suggestion = remove the `unsafe(...)`
    .note = extraneous unsafe is not allowed in attributes

attr_parsing_invalid_issue_string =
    `issue` must be a non-zero numeric string or "none"
    .must_not_be_zero = `issue` must not be "0", use "none" instead
    .empty = cannot parse integer from empty string
    .invalid_digit = invalid digit found in string
    .pos_overflow = number too large to fit in target type
    .neg_overflow = number too small to fit in target type

attr_parsing_invalid_link_modifier =
    invalid linking modifier syntax, expected '+' or '-' prefix before one of: bundle, verbatim, whole-archive, as-needed

attr_parsing_invalid_meta_item = expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found {$descr}
    .remove_neg_sugg = negative numbers are not literals, try removing the `-` sign
    .quote_ident_sugg = surround the identifier with quotation marks to make it into a string literal

attr_parsing_invalid_predicate =
    invalid predicate `{$predicate}`

attr_parsing_invalid_repr_align_need_arg =
    invalid `repr(align)` attribute: `align` needs an argument
    .suggestion = supply an argument here

attr_parsing_invalid_repr_generic =
    invalid `repr({$repr_arg})` attribute: {$error_part}

attr_parsing_invalid_repr_hint_no_paren =
    invalid representation hint: `{$name}` does not take a parenthesized argument list

attr_parsing_invalid_repr_hint_no_value =
    invalid representation hint: `{$name}` does not take a value

attr_parsing_invalid_since =
    'since' must be a Rust version number, such as "1.31.0"

attr_parsing_invalid_style = {$is_used_as_inner ->
        [false] crate-level attribute should be an inner attribute: add an exclamation mark: `#![{$name}]`
        *[other] the `#![{$name}]` attribute can only be used at the crate root
    }
    .note = This attribute does not have an `!`, which means it is applied to this {$target}

attr_parsing_invalid_target = `#[{$name}]` attribute cannot be used on {$target}
    .help = `#[{$name}]` can {$only}be applied to {$applied}
    .suggestion = remove the attribute
attr_parsing_invalid_target_lint = `#[{$name}]` attribute cannot be used on {$target}
    .warn = {-attr_parsing_previously_accepted}
    .help = `#[{$name}]` can {$only}be applied to {$applied}
    .suggestion = remove the attribute

attr_parsing_limit_invalid =
    `limit` must be a non-negative integer
    .label = {$error_str}
attr_parsing_link_arg_unstable =
    link kind `link-arg` is unstable

attr_parsing_link_cfg_unstable =
    link cfg is unstable

attr_parsing_link_framework_apple =
    link kind `framework` is only supported on Apple targets

attr_parsing_link_ordinal_out_of_range = ordinal value in `link_ordinal` is too large: `{$ordinal}`
    .note = the value may not exceed `u16::MAX`

attr_parsing_link_requires_name =
    `#[link]` attribute requires a `name = "string"` argument
    .label = missing `name` argument

attr_parsing_meta_bad_delim = wrong meta list delimiters
attr_parsing_meta_bad_delim_suggestion = the delimiters should be `(` and `)`

attr_parsing_missing_feature =
    missing 'feature'

attr_parsing_missing_issue =
    missing 'issue'

attr_parsing_missing_note =
    missing 'note'

attr_parsing_missing_since =
    missing 'since'

attr_parsing_multiple_modifiers =
    multiple `{$modifier}` modifiers in a single `modifiers` argument

attr_parsing_multiple_stability_levels =
    multiple stability levels

attr_parsing_naked_functions_incompatible_attribute =
    attribute incompatible with `#[unsafe(naked)]`
    .label = the `{$attr}` attribute is incompatible with `#[unsafe(naked)]`
    .naked_attribute = function marked with `#[unsafe(naked)]` here

attr_parsing_non_ident_feature =
    'feature' is not an identifier

attr_parsing_null_on_export = `export_name` may not contain null characters

attr_parsing_null_on_link_section = `link_section` may not contain null characters

attr_parsing_null_on_objc_class = `objc::class!` may not contain null characters

attr_parsing_null_on_objc_selector = `objc::selector!` may not contain null characters

attr_parsing_objc_class_expected_string_literal = `objc::class!` expected a string literal

attr_parsing_objc_selector_expected_string_literal = `objc::selector!` expected a string literal

attr_parsing_raw_dylib_elf_unstable =
    link kind `raw-dylib` is unstable on ELF platforms

attr_parsing_raw_dylib_no_nul =
    link name must not contain NUL characters if link kind is `raw-dylib`

attr_parsing_raw_dylib_only_windows =
    link kind `raw-dylib` is only supported on Windows targets

attr_parsing_repr_ident =
    meta item in `repr` must be an identifier

attr_parsing_rustc_allowed_unstable_pairing =
    `rustc_allowed_through_unstable_modules` attribute must be paired with a `stable` attribute

attr_parsing_rustc_promotable_pairing =
    `rustc_promotable` attribute must be paired with either a `rustc_const_unstable` or a `rustc_const_stable` attribute

attr_parsing_soft_no_args =
    `soft` should not have any arguments

attr_parsing_stability_outside_std = stability attributes may not be used outside of the standard library

attr_parsing_suffixed_literal_in_attribute = suffixed literals are not allowed in attributes
    .help = instead of using a suffixed literal (`1u8`, `1.0f32`, etc.), use an unsuffixed version (`1`, `1.0`, etc.)

attr_parsing_unknown_meta_item =
    unknown meta item '{$item}'
    .label = expected one of {$expected}

attr_parsing_unknown_version_literal =
    unknown version literal format, assuming it refers to a future version

attr_parsing_unrecognized_repr_hint =
    unrecognized representation hint
    .help = valid reprs are `Rust` (default), `C`, `align`, `packed`, `transparent`, `simd`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `i128`, `u128`, `isize`, `usize`
    .note = for more information, visit <https://doc.rust-lang.org/reference/type-layout.html?highlight=repr#representations>

attr_parsing_unsafe_attr_outside_unsafe = unsafe attribute used without unsafe
    .label = usage of unsafe attribute
attr_parsing_unsafe_attr_outside_unsafe_suggestion = wrap the attribute in `unsafe(...)`

attr_parsing_unstable_cfg_target_compact =
    compact `cfg(target(..))` is experimental and subject to change

attr_parsing_unstable_feature_bound_incompatible_stability = item annotated with `#[unstable_feature_bound]` should not be stable
    .help = If this item is meant to be stable, do not use any functions annotated with `#[unstable_feature_bound]`. Otherwise, mark this item as unstable with `#[unstable]`

attr_parsing_unsupported_literal_cfg_boolean =
    literal in `cfg` predicate value must be a boolean
attr_parsing_unsupported_literal_cfg_string =
    literal in `cfg` predicate value must be a string
attr_parsing_unsupported_literal_generic =
    unsupported literal
attr_parsing_unsupported_literal_suggestion =
    consider removing the prefix

attr_parsing_unused_duplicate =
    unused attribute
    .suggestion = remove this attribute
    .note = attribute also specified here
    .warn = {-attr_parsing_previously_accepted}

attr_parsing_unused_multiple =
    multiple `{$name}` attributes
    .suggestion = remove this attribute
    .note = attribute also specified here

-attr_parsing_previously_accepted =
    this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

attr_parsing_whole_archive_needs_static =
    linking modifier `whole-archive` is only compatible with `static` linking kind
