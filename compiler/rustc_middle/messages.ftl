middle_adjust_for_foreign_abi_error =
    target architecture {$arch} does not support `extern {$abi}` ABI

middle_assert_async_resume_after_panic = `async fn` resumed after panicking

middle_assert_async_resume_after_return = `async fn` resumed after completion

middle_assert_divide_by_zero =
    attempt to divide `{$val}` by zero

middle_assert_generator_resume_after_panic = generator resumed after panicking

middle_assert_generator_resume_after_return = generator resumed after completion

middle_assert_misaligned_ptr_deref =
    misaligned pointer dereference: address must be a multiple of {$required} but is {$found}

middle_assert_op_overflow =
    attempt to compute `{$left} {$op} {$right}`, which would overflow

middle_assert_overflow_neg =
    attempt to negate `{$val}`, which would overflow

middle_assert_remainder_by_zero =
    attempt to calculate the remainder of `{$val}` with a divisor of zero

middle_assert_shl_overflow =
    attempt to shift left by `{$val}`, which would overflow

middle_assert_shr_overflow =
    attempt to shift right by `{$val}`, which would overflow

middle_bounds_check =
    index out of bounds: the length is {$len} but the index is {$index}

middle_cannot_be_normalized =
    unable to determine layout for `{$ty}` because `{$failure_ty}` cannot be normalized

middle_conflict_types =
    this expression supplies two conflicting concrete types for the same opaque type

middle_const_eval_non_int =
    constant evaluation of enum discriminant resulted in non-integer

middle_const_not_used_in_type_alias =
    const parameter `{$ct}` is part of concrete type but not used in parameter list for the `impl Trait` type alias

middle_cycle =
    a cycle occurred during layout computation

middle_drop_check_overflow =
    overflow while adding drop-check rules for {$ty}
    .note = overflowed on {$overflow_ty}

middle_limit_invalid =
    `limit` must be a non-negative integer
    .label = {$error_str}

middle_opaque_hidden_type_mismatch =
    concrete type differs from previous defining opaque type use
    .label = expected `{$self_ty}`, got `{$other_ty}`

middle_previous_use_here =
    previous use here

middle_recursion_limit_reached =
    reached the recursion limit finding the struct tail for `{$ty}`
    .help = consider increasing the recursion limit by adding a `#![recursion_limit = "{$suggested_limit}"]`

middle_requires_lang_item = requires `{$name}` lang_item

middle_strict_coherence_needs_negative_coherence =
    to use `strict_coherence` on this trait, the `with_negative_coherence` feature must be enabled
    .label = due to this attribute

middle_unknown_layout =
    the type `{$ty}` has an unknown layout

middle_values_too_big =
    values of the type `{$ty}` are too big for the current architecture
