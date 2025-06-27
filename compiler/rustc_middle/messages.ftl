middle_assert_async_resume_after_drop = `async fn` resumed after async drop

middle_assert_async_resume_after_panic = `async fn` resumed after panicking

middle_assert_async_resume_after_return = `async fn` resumed after completion

middle_assert_coroutine_resume_after_drop = coroutine resumed after async drop

middle_assert_coroutine_resume_after_panic = coroutine resumed after panicking

middle_assert_coroutine_resume_after_return = coroutine resumed after completion

middle_assert_divide_by_zero =
    attempt to divide `{$val}` by zero

middle_assert_gen_resume_after_drop = `gen` fn or block cannot be further iterated on after it async dropped

middle_assert_gen_resume_after_panic = `gen` fn or block cannot be further iterated on after it panicked

middle_assert_misaligned_ptr_deref =
    misaligned pointer dereference: address must be a multiple of {$required} but is {$found}

middle_assert_null_ptr_deref =
    null pointer dereference occurred

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

middle_autodiff_unsafe_inner_const_ref = reading from a `Duplicated` const {$ty} is unsafe

middle_bounds_check =
    index out of bounds: the length is {$len} but the index is {$index}

middle_conflict_types =
    this expression supplies two conflicting concrete types for the same opaque type

middle_consider_type_length_limit =
    consider adding a `#![type_length_limit="{$type_length}"]` attribute to your crate

middle_const_eval_non_int =
    constant evaluation of enum discriminant resulted in non-integer

middle_const_not_used_in_type_alias =
    const parameter `{$ct}` is part of concrete type but not used in parameter list for the `impl Trait` type alias

middle_deprecated = use of deprecated {$kind} `{$path}`{$has_note ->
        [true] : {$note}
        *[other] {""}
    }
middle_deprecated_in_future = use of {$kind} `{$path}` that will be deprecated in a future Rust version{$has_note ->
        [true] : {$note}
        *[other] {""}
    }
middle_deprecated_in_version = use of {$kind} `{$path}` that will be deprecated in future version {$version}{$has_note ->
        [true] : {$note}
        *[other] {""}
    }
middle_deprecated_suggestion = replace the use of the deprecated {$kind}

middle_drop_check_overflow =
    overflow while adding drop-check rules for `{$ty}`
    .note = overflowed on `{$overflow_ty}`

middle_erroneous_constant = erroneous constant encountered

middle_failed_writing_file =
    failed to write file {$path}: {$error}"

# Note: We only mention patterns here since the error can only occur with references, and those
# are forbidden in const generics.
middle_invalid_const_in_valtree = constant {$global_const_id} cannot be used as pattern
    .note = constants that reference mutable or external memory cannot be used as pattern

middle_layout_cycle =
    a cycle occurred during layout computation

middle_layout_normalization_failure =
    unable to determine layout for `{$ty}` because `{$failure_ty}` cannot be normalized

middle_layout_references_error =
    the type has an unknown layout

middle_layout_size_overflow =
    values of the type `{$ty}` are too big for the target architecture

middle_layout_too_generic = the type `{$ty}` does not have a fixed layout

middle_layout_unknown =
    the type `{$ty}` has an unknown layout

middle_max_num_nodes_in_valtree = maximum number of nodes exceeded in constant {$global_const_id}

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

middle_type_length_limit = reached the type-length limit while instantiating `{$shrunk}`

middle_unsupported_union = we don't support unions yet: '{$ty_name}'

middle_written_to_path = the full type name has been written to '{$path}'
