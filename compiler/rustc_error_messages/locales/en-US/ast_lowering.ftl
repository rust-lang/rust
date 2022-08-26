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

ast_lowering_inline_asm_unsupported_target =
    inline assembly is unsupported on this target

ast_lowering_att_syntax_only_x86 =
    the `att_syntax` option is only supported on x86

ast_lowering_abi_specified_multiple_times =
    `{$prev_name}` ABI specified multiple times
    .label = previously specified here
    .note = these ABIs are equivalent on the current target

ast_lowering_clobber_abi_not_supported =
    `clobber_abi` is not supported on this target

ast_lowering_invalid_abi_clobber_abi =
    invalid ABI for `clobber_abi`
    .note = the following ABIs are supported on this target: {$supported_abis}

ast_lowering_invalid_register =
    invalid register `{$reg}`: {$error}

ast_lowering_invalid_register_class =
    invalid register class `{$reg_class}`: {$error}

ast_lowering_invalid_asm_template_modifier_reg_class =
    invalid asm template modifier for this register class

ast_lowering_argument = argument

ast_lowering_template_modifier = template modifier

ast_lowering_support_modifiers =
    the `{$class_name}` register class supports the following template modifiers: {$modifiers}

ast_lowering_does_not_support_modifiers =
    the `{$class_name}` register class does not support template modifiers

ast_lowering_invalid_asm_template_modifier_const =
    asm template modifiers are not allowed for `const` arguments

ast_lowering_invalid_asm_template_modifier_sym =
    asm template modifiers are not allowed for `sym` arguments

ast_lowering_register_class_only_clobber =
    register class `{$reg_class_name}` can only be used as a clobber, not as an input or output

ast_lowering_register_conflict =
    register `{$reg1_name}` conflicts with register `{$reg2_name}`
    .help = use `lateout` instead of `out` to avoid conflict

ast_lowering_register1 = register `{$reg1_name}`

ast_lowering_register2 = register `{$reg2_name}`

ast_lowering_sub_tuple_binding =
    `{$ident_name} @` is not allowed in a {$ctx}
    .label = this is only allowed in slice patterns
    .help = remove this and bind each tuple field independently

ast_lowering_sub_tuple_binding_suggestion = if you don't need to use the contents of {$ident}, discard the tuple's remaining fields

ast_lowering_extra_double_dot =
    `..` can only be used once per {$ctx} pattern
    .label = can only be used once per {$ctx} pattern

ast_lowering_previously_used_here = previously used here

ast_lowering_misplaced_double_dot =
    `..` patterns are not allowed here
    .note = only allowed in tuple, tuple struct, and slice patterns

ast_lowering_misplaced_relax_trait_bound =
    `?Trait` bounds are only permitted at the point where a type parameter is declared

ast_lowering_not_supported_for_lifetime_binder_async_closure =
    `for<...>` binders on `async` closures are not currently supported

ast_lowering_arbitrary_expression_in_pattern =
    arbitrary expressions aren't allowed in patterns
