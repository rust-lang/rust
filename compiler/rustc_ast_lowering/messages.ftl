ast_lowering_abi_specified_multiple_times =
    `{$prev_name}` ABI specified multiple times
    .label = previously specified here
    .note = these ABIs are equivalent on the current target

ast_lowering_arbitrary_expression_in_pattern =
    arbitrary expressions aren't allowed in patterns
    .pattern_from_macro_note = the `expr` fragment specifier forces the metavariable's content to be an expression

ast_lowering_argument = argument

ast_lowering_assoc_ty_binding_in_dyn =
    associated type bounds are not allowed in `dyn` types
    .suggestion = use `impl Trait` to introduce a type instead

ast_lowering_assoc_ty_parentheses =
    parenthesized generic arguments cannot be used in associated type constraints

ast_lowering_async_bound_not_on_trait =
    `async` bound modifier only allowed on trait, not `{$descr}`

ast_lowering_async_bound_only_for_fn_traits =
    `async` bound modifier only allowed on `Fn`/`FnMut`/`FnOnce` traits

ast_lowering_async_coroutines_not_supported =
    `async` coroutines are not yet supported

ast_lowering_att_syntax_only_x86 =
    the `att_syntax` option is only supported on x86

ast_lowering_await_only_in_async_fn_and_blocks =
    `await` is only allowed inside `async` functions and blocks
    .label = only allowed inside `async` functions and blocks

ast_lowering_bad_return_type_notation_inputs =
    argument types not allowed with return type notation
    .suggestion = remove the input types

ast_lowering_bad_return_type_notation_needs_dots = return type notation arguments must be elided with `..`
    .suggestion = use the correct syntax by adding `..` to the arguments

ast_lowering_bad_return_type_notation_output =
    return type not allowed with return type notation
ast_lowering_bad_return_type_notation_output_suggestion = use the right argument notation and remove the return type

ast_lowering_bad_return_type_notation_position = return type notation not allowed in this position yet

ast_lowering_clobber_abi_not_supported =
    `clobber_abi` is not supported on this target

ast_lowering_closure_cannot_be_static = closures cannot be static

ast_lowering_coroutine_too_many_parameters =
    too many parameters for a coroutine (expected 0 or 1 parameters)

ast_lowering_default_field_in_tuple = default fields are not supported in tuple structs
    .label = default fields are only supported on structs

ast_lowering_does_not_support_modifiers =
    the `{$class_name}` register class does not support template modifiers

ast_lowering_extra_double_dot =
    `..` can only be used once per {$ctx} pattern
    .label = can only be used once per {$ctx} pattern

ast_lowering_functional_record_update_destructuring_assignment =
    functional record updates are not allowed in destructuring assignments
    .suggestion = consider removing the trailing pattern

ast_lowering_generic_param_default_in_binder =
    defaults for generic parameters are not allowed in `for<...>` binders

ast_lowering_generic_type_with_parentheses =
    parenthesized type parameters may only be used with a `Fn` trait
    .label = only `Fn` traits may use parentheses

ast_lowering_inclusive_range_with_no_end = inclusive range with no end

ast_lowering_inline_asm_unsupported_target =
    inline assembly is unsupported on this target

ast_lowering_invalid_abi =
    invalid ABI: found `{$abi}`
    .label = invalid ABI
    .note = invoke `{$command}` for a full list of supported calling conventions

ast_lowering_invalid_abi_clobber_abi =
    invalid ABI for `clobber_abi`
    .note = the following ABIs are supported on this target: {$supported_abis}

ast_lowering_invalid_abi_suggestion = there's a similarly named valid ABI `{$suggestion}`

ast_lowering_invalid_asm_template_modifier_const =
    asm template modifiers are not allowed for `const` arguments

ast_lowering_invalid_asm_template_modifier_label =
    asm template modifiers are not allowed for `label` arguments

ast_lowering_invalid_asm_template_modifier_reg_class =
    invalid asm template modifier for this register class

ast_lowering_invalid_asm_template_modifier_sym =
    asm template modifiers are not allowed for `sym` arguments

ast_lowering_invalid_legacy_const_generic_arg =
    invalid argument to a legacy const generic: cannot have const blocks, closures, async blocks or items

ast_lowering_invalid_legacy_const_generic_arg_suggestion =
    try using a const generic argument instead

ast_lowering_invalid_register =
    invalid register `{$reg}`: {$error}

ast_lowering_invalid_register_class =
    invalid register class `{$reg_class}`: unknown register class
    .note = the following register classes are supported on this target: {$supported_register_classes}

ast_lowering_match_arm_with_no_body =
    `match` arm with no body
    .suggestion = add a body after the pattern

ast_lowering_misplaced_double_dot =
    `..` patterns are not allowed here
    .note = only allowed in tuple, tuple struct, and slice patterns

ast_lowering_misplaced_impl_trait =
    `impl Trait` is not allowed in {$position}
    .note = `impl Trait` is only allowed in arguments and return types of functions and methods

ast_lowering_never_pattern_with_body =
    a never pattern is always unreachable
    .label = this will never be executed
    .suggestion = remove this expression

ast_lowering_never_pattern_with_guard =
    a guard on a never pattern will never be run
    .suggestion = remove this guard

ast_lowering_no_precise_captures_on_apit = `use<...>` precise capturing syntax not allowed in argument-position `impl Trait`

ast_lowering_previously_used_here = previously used here

ast_lowering_register1 = register `{$reg1_name}`

ast_lowering_register2 = register `{$reg2_name}`

ast_lowering_register_class_only_clobber =
    register class `{$reg_class_name}` can only be used as a clobber, not as an input or output
ast_lowering_register_class_only_clobber_stable =
    register class `{$reg_class_name}` can only be used as a clobber in stable

ast_lowering_register_conflict =
    register `{$reg1_name}` conflicts with register `{$reg2_name}`
    .help = use `lateout` instead of `out` to avoid conflict

ast_lowering_remove_parentheses = remove these parentheses

ast_lowering_sub_tuple_binding =
    `{$ident_name} @` is not allowed in a {$ctx}
    .label = this is only allowed in slice patterns
    .help = remove this and bind each tuple field independently

ast_lowering_sub_tuple_binding_suggestion = if you don't need to use the contents of {$ident}, discard the tuple's remaining fields

ast_lowering_support_modifiers =
    the `{$class_name}` register class supports the following template modifiers: {$modifiers}

ast_lowering_template_modifier = template modifier

ast_lowering_this_not_async = this is not `async`

ast_lowering_underscore_expr_lhs_assign =
    in expressions, `_` can only be used on the left-hand side of an assignment
    .label = `_` not allowed here

ast_lowering_union_default_field_values = unions cannot have default field values

ast_lowering_unstable_inline_assembly = inline assembly is not stable yet on this architecture
ast_lowering_unstable_inline_assembly_label_operand_with_outputs =
    using both label and output operands for inline assembly is unstable
ast_lowering_unstable_may_unwind = the `may_unwind` option is unstable

ast_lowering_use_angle_brackets = use angle brackets instead

ast_lowering_yield = yield syntax is experimental
ast_lowering_yield_in_closure =
    `yield` can only be used in `#[coroutine]` closures, or `gen` blocks
    .suggestion = use `#[coroutine]` to make this closure a coroutine
