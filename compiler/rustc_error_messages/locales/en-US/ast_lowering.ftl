ast_lowering_generic_type_with_parentheses =
    parenthesized type parameters may only be used with a `Fn` trait
    .label = only `Fn` traits may use parentheses

ast_lowering_use_angle_brackets = use angle brackets instead

ast_lowering_invalid_abi =
    invalid ABI: found `{$abi}`
    .label = invalid ABI
    .help = valid ABIs: {$valid_abis}

ast_lowering_assoc_ty_parentheses =
    parenthesized generic arguments cannot be used in associated type constraints

ast_lowering_remove_parentheses = remove these parentheses

ast_lowering_misplaced_impl_trait =
    `impl Trait` only allowed in function and inherent method return types, not in {$position}
