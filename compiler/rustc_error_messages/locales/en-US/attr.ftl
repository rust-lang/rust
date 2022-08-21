attr_multiple_item =
    multiple '{$item}' items

attr_unknown_meta_item =
    unknown meta item '{$item}'
    .label = expected one of {$expected}

attr_missing_since =
    missing 'since'

attr_non_ident_feature =
    'feature' is not an identifier

attr_missing_feature =
    missing 'feature'

attr_multiple_stability_levels =
    multiple stability levels

attr_unsupported_literal_generic =
    unsupported literal
attr_unsupported_literal_cfg_string =
    literal in `cfg` predicate value must be a string
attr_unsupported_literal_deprecated_string =
    literal in `deprecated` value must be a string
attr_unsupported_literal_deprecated_kv_pair =
    item in `deprecated` must be a key/value pair
attr_unsupported_literal_suggestion =
    consider removing the prefix

attr_invalid_meta_item =
    incorrect meta item

attr_invalid_issue_string =
    `issue` must be a non-zero numeric string or "none"
attr_must_not_be_zero =
    `issue` must not be "0", use "none" instead
attr_empty =
    cannot parse integer from empty string
attr_invalid_digit =
    invalid digit found in string
attr_pos_overflow =
    number too large to fit in target type
attr_neg_overflow =
    number too small to fit in target type

attr_missing_issue =
    missing 'issue'

attr_rustc_promotable_pairing =
    `rustc_promotable` attribute must be paired with either a `rustc_const_unstable` or a `rustc_const_stable` attribute

attr_rustc_allowed_unstable_pairing =
    `rustc_allowed_through_unstable_modules` attribute must be paired with a `stable` attribute

attr_soft_no_args =
    `soft` should not have any arguments
