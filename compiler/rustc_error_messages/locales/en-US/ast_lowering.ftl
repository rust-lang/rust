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

ast_lowering_rustc_box_attribute_error =
    #[rustc_box] requires precisely one argument and no other attributes are allowed

ast_lowering_underscore_expr_lhs_assign =
    in expressions, `_` can only be used on the left-hand side of an assignment
    .label = `_` not allowed here

ast_lowering_base_expression_double_dot =
    base expression required after `..`
    .label = add a base expression here

ast_lowering_await_only_in_async_fn_and_blocks =
    `await` is only allowed inside `async` functions and blocks
    .label = only allowed inside `async` functions and blocks

ast_lowering_this_not_async = this is not `async`

ast_lowering_generator_too_many_parameters =
    too many parameters for a generator (expected 0 or 1 parameters)

ast_lowering_closure_cannot_be_static = closures cannot be static

ast_lowering_async_non_move_closure_not_supported =
    `async` non-`move` closures with parameters are not currently supported
    .help = consider using `let` statements to manually capture variables by reference before entering an `async move` closure

ast_lowering_functional_record_update_destructuring_assignment =
    functional record updates are not allowed in destructuring assignments
    .suggestion = consider removing the trailing pattern

ast_lowering_async_generators_not_supported =
    `async` generators are not yet supported
