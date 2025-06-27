attr_parsing_cfg_predicate_identifier =
    `cfg` predicate key must be an identifier

attr_parsing_deprecated_item_suggestion =
    suggestions on deprecated items are unstable
    .help = add `#![feature(deprecated_suggestion)]` to the crate root
    .note = see #94785 for more details

attr_parsing_empty_confusables =
    expected at least one confusable name
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

attr_parsing_invalid_issue_string =
    `issue` must be a non-zero numeric string or "none"
    .must_not_be_zero = `issue` must not be "0", use "none" instead
    .empty = cannot parse integer from empty string
    .invalid_digit = invalid digit found in string
    .pos_overflow = number too large to fit in target type
    .neg_overflow = number too small to fit in target type

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

attr_parsing_missing_feature =
    missing 'feature'

attr_parsing_missing_issue =
    missing 'issue'

attr_parsing_missing_note =
    missing 'note'

attr_parsing_missing_since =
    missing 'since'

attr_parsing_multiple_stability_levels =
    multiple stability levels

attr_parsing_naked_functions_incompatible_attribute =
    attribute incompatible with `#[unsafe(naked)]`
    .label = the `{$attr}` attribute is incompatible with `#[unsafe(naked)]`
    .naked_attribute = function marked with `#[unsafe(naked)]` here

attr_parsing_non_ident_feature =
    'feature' is not an identifier

attr_parsing_null_on_export = `export_name` may not contain null characters

attr_parsing_repr_ident =
    meta item in `repr` must be an identifier

attr_parsing_rustc_allowed_unstable_pairing =
    `rustc_allowed_through_unstable_modules` attribute must be paired with a `stable` attribute

attr_parsing_rustc_promotable_pairing =
    `rustc_promotable` attribute must be paired with either a `rustc_const_unstable` or a `rustc_const_stable` attribute

attr_parsing_soft_no_args =
    `soft` should not have any arguments

attr_parsing_stability_outside_std = stability attributes may not be used outside of the standard library

attr_parsing_unknown_meta_item =
    unknown meta item '{$item}'
    .label = expected one of {$expected}

attr_parsing_unknown_version_literal =
    unknown version literal format, assuming it refers to a future version

attr_parsing_unrecognized_repr_hint =
    unrecognized representation hint
    .help = valid reprs are `Rust` (default), `C`, `align`, `packed`, `transparent`, `simd`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `i128`, `u128`, `isize`, `usize`

attr_parsing_unstable_cfg_target_compact =
    compact `cfg(target(..))` is experimental and subject to change

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
    .warn = {-passes_previously_accepted}

attr_parsing_unused_multiple =
    multiple `{$name}` attributes
    .suggestion = remove this attribute
    .note = attribute also specified here

-attr_parsing_perviously_accepted =
    this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
