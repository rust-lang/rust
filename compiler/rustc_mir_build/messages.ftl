mir_build_adt_defined_here = `{$ty}` defined here

mir_build_already_borrowed = cannot borrow value as mutable because it is also borrowed as immutable

mir_build_already_mut_borrowed = cannot borrow value as immutable because it is also borrowed as mutable

mir_build_bindings_with_variant_name =
    pattern binding `{$name}` is named the same as one of the variants of the type `{$ty_path}`
    .suggestion = to match on the variant, qualify the path

mir_build_borrow = value is borrowed by `{$name}` here

mir_build_borrow_of_layout_constrained_field_requires_unsafe =
    borrow of layout constrained field with interior mutability is unsafe and requires unsafe block
    .note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
    .label = borrow of layout constrained field with interior mutability

mir_build_borrow_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    borrow of layout constrained field with interior mutability is unsafe and requires unsafe function or block
    .note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
    .label = borrow of layout constrained field with interior mutability

mir_build_borrow_of_moved_value = borrow of moved value
    .label = value moved into `{$name}` here
    .occurs_because_label = move occurs because `{$name}` has type `{$ty}`, which does not implement the `Copy` trait
    .value_borrowed_label = value borrowed here after move
    .suggestion = borrow this binding in the pattern to avoid moving the value

mir_build_call_to_deprecated_safe_fn_requires_unsafe =
    call to deprecated safe function `{$function}` is unsafe and requires unsafe block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function
    .suggestion = you can wrap the call in an `unsafe` block if you can guarantee {$guarantee}

mir_build_call_to_fn_with_requires_unsafe =
    call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe block
    .help = in order for the call to be safe, the context requires the following additional target {$missing_target_features_count ->
        [1] feature
        *[count] features
        }: {$missing_target_features}
    .note = the {$build_target_features} target {$build_target_features_count ->
        [1] feature
        *[count] features
        } being enabled in the build configuration does not remove the requirement to list {$build_target_features_count ->
        [1] it
        *[count] them
        } in `#[target_feature]`
    .label = call to function with `#[target_feature]`

mir_build_call_to_fn_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe function or block
    .help = in order for the call to be safe, the context requires the following additional target {$missing_target_features_count ->
        [1] feature
        *[count] features
        }: {$missing_target_features}
    .note = the {$build_target_features} target {$build_target_features_count ->
        [1] feature
        *[count] features
        } being enabled in the build configuration does not remove the requirement to list {$build_target_features_count ->
        [1] it
        *[count] them
        } in `#[target_feature]`
    .label = call to function with `#[target_feature]`

mir_build_call_to_unsafe_fn_requires_unsafe =
    call to unsafe function `{$function}` is unsafe and requires unsafe block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_call_to_unsafe_fn_requires_unsafe_nameless =
    call to unsafe function is unsafe and requires unsafe block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_call_to_unsafe_fn_requires_unsafe_nameless_unsafe_op_in_unsafe_fn_allowed =
    call to unsafe function is unsafe and requires unsafe function or block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_call_to_unsafe_fn_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    call to unsafe function `{$function}` is unsafe and requires unsafe function or block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_confused = missing patterns are not covered because `{$variable}` is interpreted as a constant pattern, not a new variable

mir_build_const_continue_bad_const = could not determine the target branch for this `#[const_continue]`
    .label = this value is too generic

mir_build_const_continue_missing_label_or_value = a `#[const_continue]` must break to a label with a value

mir_build_const_continue_not_const = could not determine the target branch for this `#[const_continue]`
    .help = try extracting the expression into a `const` item

mir_build_const_continue_not_const_const_block = `const` blocks may use generics, and are not evaluated early enough
mir_build_const_continue_not_const_const_other = this value must be a literal or a monomorphic const
mir_build_const_continue_not_const_constant_parameter = constant parameters may use generics, and are not evaluated early enough

mir_build_const_continue_unknown_jump_target = the target of this `#[const_continue]` is not statically known
    .label = this value must be a literal or a monomorphic const

mir_build_const_defined_here = constant defined here

mir_build_const_param_in_pattern = constant parameters cannot be referenced in patterns
    .label = can't be used in patterns
mir_build_const_param_in_pattern_def = constant defined here

mir_build_const_pattern_depends_on_generic_parameter = constant pattern cannot depend on generic parameters
    .label = `const` depends on a generic parameter

mir_build_could_not_eval_const_pattern = could not evaluate constant pattern
    .label = could not evaluate constant

mir_build_deref_raw_pointer_requires_unsafe =
    dereference of raw pointer is unsafe and requires unsafe block
    .note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
    .label = dereference of raw pointer

mir_build_deref_raw_pointer_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    dereference of raw pointer is unsafe and requires unsafe function or block
    .note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
    .label = dereference of raw pointer

mir_build_extern_static_requires_unsafe =
    use of extern static is unsafe and requires unsafe block
    .note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
    .label = use of extern static

mir_build_extern_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of extern static is unsafe and requires unsafe function or block
    .note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
    .label = use of extern static

mir_build_inform_irrefutable = `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with only one variant

mir_build_initializing_type_with_requires_unsafe =
    initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe block
    .note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
    .label = initializing type with `rustc_layout_scalar_valid_range` attr

mir_build_initializing_type_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe function or block
    .note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
    .label = initializing type with `rustc_layout_scalar_valid_range` attr

mir_build_initializing_type_with_unsafe_field_requires_unsafe =
    initializing type with an unsafe field is unsafe and requires unsafe block
    .note = unsafe fields may carry library invariants
    .label = initialization of struct with unsafe field

mir_build_initializing_type_with_unsafe_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    initializing type with an unsafe field is unsafe and requires unsafe block
    .note = unsafe fields may carry library invariants
    .label = initialization of struct with unsafe field

mir_build_inline_assembly_requires_unsafe =
    use of inline assembly is unsafe and requires unsafe block
    .note = inline assembly is entirely unchecked and can cause undefined behavior
    .label = use of inline assembly

mir_build_inline_assembly_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of inline assembly is unsafe and requires unsafe function or block
    .note = inline assembly is entirely unchecked and can cause undefined behavior
    .label = use of inline assembly

mir_build_interpreted_as_const = introduce a variable instead

mir_build_invalid_pattern = {$prefix} `{$non_sm_ty}` cannot be used in patterns
    .label = {$prefix} can't be used in patterns

mir_build_irrefutable_let_patterns_if_let = irrefutable `if let` {$count ->
        [one] pattern
        *[other] patterns
    }
    .note = {$count ->
        [one] this pattern
        *[other] these patterns
    } will always match, so the `if let` is useless
    .help = consider replacing the `if let` with a `let`

mir_build_irrefutable_let_patterns_if_let_guard = irrefutable `if let` guard {$count ->
        [one] pattern
        *[other] patterns
    }
    .note = {$count ->
        [one] this pattern
        *[other] these patterns
    } will always match, so the guard is useless
    .help = consider removing the guard and adding a `let` inside the match arm

mir_build_irrefutable_let_patterns_let_else = irrefutable `let...else` {$count ->
        [one] pattern
        *[other] patterns
    }
    .note = {$count ->
        [one] this pattern
        *[other] these patterns
    } will always match, so the `else` clause is useless
    .help = consider removing the `else` clause

mir_build_irrefutable_let_patterns_while_let = irrefutable `while let` {$count ->
        [one] pattern
        *[other] patterns
    }
    .note = {$count ->
        [one] this pattern
        *[other] these patterns
    } will always match, so the loop will never exit
    .help = consider instead using a `loop {"{"} ... {"}"}` with a `let` inside it

mir_build_leading_irrefutable_let_patterns = leading irrefutable {$count ->
        [one] pattern
        *[other] patterns
    } in let chain
    .note = {$count ->
        [one] this pattern
        *[other] these patterns
    } will always match
    .help = consider moving {$count ->
        [one] it
        *[other] them
    } outside of the construct

mir_build_literal_in_range_out_of_bounds =
    literal out of range for `{$ty}`
    .label = this value does not fit into the type `{$ty}` whose range is `{$min}..={$max}`

mir_build_loop_match_arm_with_guard =
    match arms that are part of a `#[loop_match]` cannot have guards

mir_build_loop_match_bad_rhs =
    this expression must be a single `match` wrapped in a labeled block

mir_build_loop_match_bad_statements =
    statements are not allowed in this position within a `#[loop_match]`

mir_build_loop_match_invalid_match =
    invalid match on `#[loop_match]` state
    .note = a local variable must be the scrutinee within a `#[loop_match]`

mir_build_loop_match_invalid_update =
    invalid update of the `#[loop_match]` state
    .label = the assignment must update this variable

mir_build_loop_match_missing_assignment =
    expected a single assignment expression

mir_build_loop_match_unsupported_type =
    this `#[loop_match]` state value has type `{$ty}`, which is not supported
    .note = only integers, floats, bool, char, and enums without fields are supported

mir_build_lower_range_bound_must_be_less_than_or_equal_to_upper =
    lower range bound must be less than or equal to upper
    .label = lower bound larger than upper bound
    .teach_note = When matching against a range, the compiler verifies that the range is non-empty. Range patterns include both end-points, so this is equivalent to requiring the start of the range to be less than or equal to the end of the range.

mir_build_lower_range_bound_must_be_less_than_upper = lower range bound must be less than upper

mir_build_more_information = for more information, visit https://doc.rust-lang.org/book/ch19-02-refutability.html

mir_build_moved = value is moved into `{$name}` here

mir_build_moved_while_borrowed = cannot move out of value because it is borrowed

mir_build_multiple_mut_borrows = cannot borrow value as mutable more than once at a time

mir_build_mutable_borrow = value is mutably borrowed by `{$name}` here

mir_build_mutable_static_requires_unsafe =
    use of mutable static is unsafe and requires unsafe block
    .note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
    .label = use of mutable static

mir_build_mutable_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of mutable static is unsafe and requires unsafe function or block
    .note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
    .label = use of mutable static

mir_build_mutation_of_layout_constrained_field_requires_unsafe =
    mutation of layout constrained field is unsafe and requires unsafe block
    .note = mutating layout constrained fields cannot statically be checked for valid values
    .label = mutation of layout constrained field

mir_build_mutation_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    mutation of layout constrained field is unsafe and requires unsafe function or block
    .note = mutating layout constrained fields cannot statically be checked for valid values
    .label = mutation of layout constrained field

mir_build_nan_pattern = cannot use NaN in patterns
    .label = evaluates to `NaN`, which is not allowed in patterns
    .note = NaNs compare inequal to everything, even themselves, so this pattern would never match
    .help = try using the `is_nan` method instead

mir_build_non_const_path = runtime values cannot be referenced in patterns
    .label = references a runtime value

mir_build_non_empty_never_pattern =
    mismatched types
    .label = a never pattern must be used on an uninhabited type
    .note = the matched value is of type `{$ty}`

mir_build_non_exhaustive_match_all_arms_guarded =
    match arms with guards don't count towards exhaustivity

mir_build_non_exhaustive_patterns_type_not_empty = non-exhaustive patterns: type `{$ty}` is non-empty
    .def_note = `{$peeled_ty}` defined here
    .type_note = the matched value is of type `{$ty}`
    .non_exhaustive_type_note = the matched value is of type `{$ty}`, which is marked as non-exhaustive
    .reference_note = references are always considered inhabited
    .suggestion = ensure that all possible cases are being handled by adding a match arm with a wildcard pattern as shown
    .help = ensure that all possible cases are being handled by adding a match arm with a wildcard pattern

mir_build_non_partial_eq_match = constant of non-structural type `{$ty}` in a pattern
    .label = constant of non-structural type

mir_build_pattern_not_covered = refutable pattern in {$origin}
    .pattern_ty = the matched value is of type `{$pattern_ty}`

mir_build_pointer_pattern = function pointers and raw pointers not derived from integers in patterns behave unpredictably and should not be relied upon
    .label = can't be used in patterns
    .note = see https://github.com/rust-lang/rust/issues/70861 for details

mir_build_privately_uninhabited = pattern `{$witness_1}` is currently uninhabited, but this variant contains private fields which may become inhabited in the future

mir_build_rust_2024_incompatible_pat = {$bad_modifiers ->
        *[true] binding modifiers{$bad_ref_pats ->
            *[true] {" "}and reference patterns
            [false] {""}
        }
        [false] reference patterns
    } may only be written when the default binding mode is `move`{$is_hard_error ->
        *[true] {""}
        [false] {" "}in Rust 2024
    }

mir_build_static_in_pattern = statics cannot be referenced in patterns
    .label = can't be used in patterns
mir_build_static_in_pattern_def = `static` defined here

mir_build_suggest_attempted_int_lit = alternatively, you could prepend the pattern with an underscore to define a new named variable; identifiers cannot begin with digits


mir_build_suggest_if_let = you might want to use `if let` to ignore the {$count ->
        [one] variant that isn't
        *[other] variants that aren't
    } matched

mir_build_suggest_let_else = you might want to use `let else` to handle the {$count ->
        [one] variant that isn't
        *[other] variants that aren't
    } matched

mir_build_trailing_irrefutable_let_patterns = trailing irrefutable {$count ->
        [one] pattern
        *[other] patterns
    } in let chain
    .note = {$count ->
        [one] this pattern
        *[other] these patterns
    } will always match
    .help = consider moving {$count ->
        [one] it
        *[other] them
    } into the body

mir_build_type_not_structural = constant of non-structural type `{$ty}` in a pattern
    .label = constant of non-structural type
mir_build_type_not_structural_def = `{$ty}` must be annotated with `#[derive(PartialEq)]` to be usable in patterns
mir_build_type_not_structural_more_info = see https://doc.rust-lang.org/stable/std/marker/trait.StructuralPartialEq.html for details
mir_build_type_not_structural_tip =
    the `PartialEq` trait must be derived, manual `impl`s are not sufficient; see https://doc.rust-lang.org/stable/std/marker/trait.StructuralPartialEq.html for details

mir_build_union_field_requires_unsafe =
    access to union field is unsafe and requires unsafe block
    .note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
    .label = access to union field

mir_build_union_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    access to union field is unsafe and requires unsafe function or block
    .note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
    .label = access to union field

mir_build_union_pattern = cannot use unions in constant patterns
    .label = can't use a `union` here

mir_build_unreachable_making_this_unreachable = collectively making this unreachable

mir_build_unreachable_making_this_unreachable_n_more = ...and {$covered_by_many_n_more_count} other patterns collectively make this unreachable

mir_build_unreachable_matches_same_values = matches some of the same values

mir_build_unreachable_pattern = unreachable pattern
    .label = no value can reach this
    .unreachable_matches_no_values = matches no values because `{$matches_no_values_ty}` is uninhabited
    .unreachable_uninhabited_note = to learn more about uninhabited types, see https://doc.rust-lang.org/nomicon/exotic-sizes.html#empty-types
    .unreachable_covered_by_catchall = matches any value
    .unreachable_covered_by_one = matches all the relevant values
    .unreachable_covered_by_many = multiple earlier patterns match some of the same values
    .unreachable_pattern_const_reexport_accessible = there is a constant of the same name imported in another scope, which could have been used to pattern match against its value instead of introducing a new catch-all binding, but it needs to be imported in the pattern's scope
    .unreachable_pattern_wanted_const = you might have meant to pattern match against the value of {$is_typo ->
        [true] similarly named constant
        *[false] constant
        } `{$const_name}` instead of introducing a new catch-all binding
    .unreachable_pattern_const_inaccessible = there is a constant of the same name, which could have been used to pattern match against its value instead of introducing a new catch-all binding, but it is not accessible from this scope
    .unreachable_pattern_let_binding = there is a binding of the same name; if you meant to pattern match against the value of that binding, that is a feature of constants that is not available for `let` bindings
    .suggestion = remove the match arm

mir_build_unsafe_binder_cast_requires_unsafe =
    unsafe binder cast is unsafe and requires unsafe block
    .label = unsafe binder cast
    .note = casting to or from an `unsafe<...>` binder type is unsafe since it erases lifetime
        information that may be required to uphold safety guarantees of a type

mir_build_unsafe_binder_cast_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    unsafe binder cast is unsafe and requires unsafe block or unsafe fn
    .label = unsafe binder cast
    .note = casting to or from an `unsafe<...>` binder type is unsafe since it erases lifetime
        information that may be required to uphold safety guarantees of a type

mir_build_unsafe_field_requires_unsafe =
    use of unsafe field is unsafe and requires unsafe block
    .note = unsafe fields may carry library invariants
    .label = use of unsafe field

mir_build_unsafe_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of unsafe field is unsafe and requires unsafe block
    .note = unsafe fields may carry library invariants
    .label = use of unsafe field

mir_build_unsafe_fn_safe_body = an unsafe function restricts its caller, but its body is safe by default
mir_build_unsafe_not_inherited = items do not inherit unsafety from separate enclosing items

mir_build_unsafe_op_in_unsafe_fn_borrow_of_layout_constrained_field_requires_unsafe =
    borrow of layout constrained field with interior mutability is unsafe and requires unsafe block
    .note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
    .label = borrow of layout constrained field with interior mutability

mir_build_unsafe_op_in_unsafe_fn_call_to_fn_with_requires_unsafe =
    call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe block
    .help = in order for the call to be safe, the context requires the following additional target {$missing_target_features_count ->
        [1] feature
        *[count] features
        }: {$missing_target_features}
    .note = the {$build_target_features} target {$build_target_features_count ->
        [1] feature
        *[count] features
        } being enabled in the build configuration does not remove the requirement to list {$build_target_features_count ->
        [1] it
        *[count] them
        } in `#[target_feature]`
    .label = call to function with `#[target_feature]`

mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe =
    call to unsafe function `{$function}` is unsafe and requires unsafe block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe_nameless =
    call to unsafe function is unsafe and requires unsafe block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_unsafe_op_in_unsafe_fn_deref_raw_pointer_requires_unsafe =
    dereference of raw pointer is unsafe and requires unsafe block
    .note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
    .label = dereference of raw pointer

mir_build_unsafe_op_in_unsafe_fn_extern_static_requires_unsafe =
    use of extern static is unsafe and requires unsafe block
    .note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
    .label = use of extern static

mir_build_unsafe_op_in_unsafe_fn_initializing_type_with_requires_unsafe =
    initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe block
    .note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
    .label = initializing type with `rustc_layout_scalar_valid_range` attr

mir_build_unsafe_op_in_unsafe_fn_initializing_type_with_unsafe_field_requires_unsafe =
    initializing type with an unsafe field is unsafe and requires unsafe block
    .note = unsafe fields may carry library invariants
    .label = initialization of struct with unsafe field

mir_build_unsafe_op_in_unsafe_fn_inline_assembly_requires_unsafe =
    use of inline assembly is unsafe and requires unsafe block
    .note = inline assembly is entirely unchecked and can cause undefined behavior
    .label = use of inline assembly

mir_build_unsafe_op_in_unsafe_fn_mutable_static_requires_unsafe =
    use of mutable static is unsafe and requires unsafe block
    .note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
    .label = use of mutable static

mir_build_unsafe_op_in_unsafe_fn_mutation_of_layout_constrained_field_requires_unsafe =
    mutation of layout constrained field is unsafe and requires unsafe block
    .note = mutating layout constrained fields cannot statically be checked for valid values
    .label = mutation of layout constrained field

mir_build_unsafe_op_in_unsafe_fn_union_field_requires_unsafe =
    access to union field is unsafe and requires unsafe block
    .note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
    .label = access to union field

mir_build_unsafe_op_in_unsafe_fn_unsafe_field_requires_unsafe =
    use of unsafe field is unsafe and requires unsafe block
    .note = unsafe fields may carry library invariants
    .label = use of unsafe field

mir_build_unsized_pattern = cannot use unsized non-slice type `{$non_sm_ty}` in constant patterns

mir_build_unused_unsafe = unnecessary `unsafe` block
    .label = unnecessary `unsafe` block

mir_build_unused_unsafe_enclosing_block_label = because it's nested under this `unsafe` block

mir_build_variant_defined_here = not covered

mir_build_wrap_suggestion = consider wrapping the function body in an unsafe block
