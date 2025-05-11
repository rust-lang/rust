pattern_analysis_excluside_range_missing_gap = multiple ranges are one apart
    .label = this range doesn't match `{$gap}` because `..` is an exclusive range
    .suggestion = use an inclusive range instead

pattern_analysis_excluside_range_missing_max = exclusive range missing `{$max}`
    .label = this range doesn't match `{$max}` because `..` is an exclusive range
    .suggestion = use an inclusive range instead

pattern_analysis_mixed_deref_pattern_constructors = mix of deref patterns and normal constructors
    .deref_pattern_label = matches on the result of dereferencing `{$smart_pointer_ty}`
    .normal_constructor_label = matches directly on `{$smart_pointer_ty}`

pattern_analysis_non_exhaustive_omitted_pattern = some variants are not matched explicitly
    .help = ensure that all variants are matched explicitly by adding the suggested match arms
    .note = the matched value is of type `{$scrut_ty}` and the `non_exhaustive_omitted_patterns` attribute was found

pattern_analysis_non_exhaustive_omitted_pattern_lint_on_arm = the lint level must be set on the whole match
    .help = it no longer has any effect to set the lint level on an individual match arm
    .label = remove this attribute
    .suggestion = set the lint level on the whole match

pattern_analysis_overlapping_range_endpoints = multiple patterns overlap on their endpoints
    .label = ... with this range
    .note = you likely meant to write mutually exclusive ranges

pattern_analysis_uncovered = {$count ->
        [1] pattern `{$witness_1}`
        [2] patterns `{$witness_1}` and `{$witness_2}`
        [3] patterns `{$witness_1}`, `{$witness_2}` and `{$witness_3}`
        *[other] patterns `{$witness_1}`, `{$witness_2}`, `{$witness_3}` and {$remainder} more
    } not covered
