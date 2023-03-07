mir_build_unconditional_recursion = function cannot return without recursing
    .label = cannot return without recursing
    .help = a `loop` may express intention better if this is on purpose

mir_build_unconditional_recursion_call_site_label = recursive call site

mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe =
    call to unsafe function `{$function}` is unsafe and requires unsafe block (error E0133)
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe_nameless =
    call to unsafe function is unsafe and requires unsafe block (error E0133)
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_unsafe_op_in_unsafe_fn_inline_assembly_requires_unsafe =
    use of inline assembly is unsafe and requires unsafe block (error E0133)
    .note = inline assembly is entirely unchecked and can cause undefined behavior
    .label = use of inline assembly

mir_build_unsafe_op_in_unsafe_fn_initializing_type_with_requires_unsafe =
    initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe
    block (error E0133)
    .note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
    .label = initializing type with `rustc_layout_scalar_valid_range` attr

mir_build_unsafe_op_in_unsafe_fn_mutable_static_requires_unsafe =
    use of mutable static is unsafe and requires unsafe block (error E0133)
    .note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
    .label = use of mutable static

mir_build_unsafe_op_in_unsafe_fn_extern_static_requires_unsafe =
    use of extern static is unsafe and requires unsafe block (error E0133)
    .note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
    .label = use of extern static

mir_build_unsafe_op_in_unsafe_fn_deref_raw_pointer_requires_unsafe =
    dereference of raw pointer is unsafe and requires unsafe block (error E0133)
    .note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
    .label = dereference of raw pointer

mir_build_unsafe_op_in_unsafe_fn_union_field_requires_unsafe =
    access to union field is unsafe and requires unsafe block (error E0133)
    .note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
    .label = access to union field

mir_build_unsafe_op_in_unsafe_fn_mutation_of_layout_constrained_field_requires_unsafe =
    mutation of layout constrained field is unsafe and requires unsafe block (error E0133)
    .note = mutating layout constrained fields cannot statically be checked for valid values
    .label = mutation of layout constrained field

mir_build_unsafe_op_in_unsafe_fn_borrow_of_layout_constrained_field_requires_unsafe =
    borrow of layout constrained field with interior mutability is unsafe and requires unsafe block (error E0133)
    .note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
    .label = borrow of layout constrained field with interior mutability

mir_build_unsafe_op_in_unsafe_fn_call_to_fn_with_requires_unsafe =
    call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe block (error E0133)
    .note = can only be called if the required target features are available
    .label = call to function with `#[target_feature]`

mir_build_call_to_unsafe_fn_requires_unsafe =
    call to unsafe function `{$function}` is unsafe and requires unsafe block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_call_to_unsafe_fn_requires_unsafe_nameless =
    call to unsafe function is unsafe and requires unsafe block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_call_to_unsafe_fn_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    call to unsafe function `{$function}` is unsafe and requires unsafe function or block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_call_to_unsafe_fn_requires_unsafe_nameless_unsafe_op_in_unsafe_fn_allowed =
    call to unsafe function is unsafe and requires unsafe function or block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_inline_assembly_requires_unsafe =
    use of inline assembly is unsafe and requires unsafe block
    .note = inline assembly is entirely unchecked and can cause undefined behavior
    .label = use of inline assembly

mir_build_inline_assembly_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of inline assembly is unsafe and requires unsafe function or block
    .note = inline assembly is entirely unchecked and can cause undefined behavior
    .label = use of inline assembly

mir_build_initializing_type_with_requires_unsafe =
    initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe block
    .note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
    .label = initializing type with `rustc_layout_scalar_valid_range` attr

mir_build_initializing_type_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe function or block
    .note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
    .label = initializing type with `rustc_layout_scalar_valid_range` attr

mir_build_mutable_static_requires_unsafe =
    use of mutable static is unsafe and requires unsafe block
    .note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
    .label = use of mutable static

mir_build_mutable_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of mutable static is unsafe and requires unsafe function or block
    .note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
    .label = use of mutable static

mir_build_extern_static_requires_unsafe =
    use of extern static is unsafe and requires unsafe block
    .note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
    .label = use of extern static

mir_build_extern_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of extern static is unsafe and requires unsafe function or block
    .note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
    .label = use of extern static

mir_build_deref_raw_pointer_requires_unsafe =
    dereference of raw pointer is unsafe and requires unsafe block
    .note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
    .label = dereference of raw pointer

mir_build_deref_raw_pointer_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    dereference of raw pointer is unsafe and requires unsafe function or block
    .note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
    .label = dereference of raw pointer

mir_build_union_field_requires_unsafe =
    access to union field is unsafe and requires unsafe block
    .note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
    .label = access to union field

mir_build_union_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    access to union field is unsafe and requires unsafe function or block
    .note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
    .label = access to union field

mir_build_mutation_of_layout_constrained_field_requires_unsafe =
    mutation of layout constrained field is unsafe and requires unsafe block
    .note = mutating layout constrained fields cannot statically be checked for valid values
    .label = mutation of layout constrained field

mir_build_mutation_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    mutation of layout constrained field is unsafe and requires unsafe function or block
    .note = mutating layout constrained fields cannot statically be checked for valid values
    .label = mutation of layout constrained field

mir_build_borrow_of_layout_constrained_field_requires_unsafe =
    borrow of layout constrained field with interior mutability is unsafe and requires unsafe block
    .note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
    .label = borrow of layout constrained field with interior mutability

mir_build_borrow_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    borrow of layout constrained field with interior mutability is unsafe and requires unsafe function or block
    .note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
    .label = borrow of layout constrained field with interior mutability

mir_build_call_to_fn_with_requires_unsafe =
    call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe block
    .note = can only be called if the required target features are available
    .label = call to function with `#[target_feature]`

mir_build_call_to_fn_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe function or block
    .note = can only be called if the required target features are available
    .label = call to function with `#[target_feature]`

mir_build_unused_unsafe = unnecessary `unsafe` block
    .label = unnecessary `unsafe` block

mir_build_unused_unsafe_enclosing_block_label = because it's nested under this `unsafe` block
mir_build_unused_unsafe_enclosing_fn_label = because it's nested under this `unsafe` fn

mir_build_non_exhaustive_patterns_type_not_empty = non-exhaustive patterns: type `{$ty}` is non-empty
    .def_note = `{$peeled_ty}` defined here
    .type_note = the matched value is of type `{$ty}`
    .non_exhaustive_type_note = the matched value is of type `{$ty}`, which is marked as non-exhaustive
    .reference_note = references are always considered inhabited
    .suggestion = ensure that all possible cases are being handled by adding a match arm with a wildcard pattern as shown
    .help = ensure that all possible cases are being handled by adding a match arm with a wildcard pattern

mir_build_static_in_pattern = statics cannot be referenced in patterns

mir_build_assoc_const_in_pattern = associated consts cannot be referenced in patterns

mir_build_const_param_in_pattern = const parameters cannot be referenced in patterns

mir_build_non_const_path = runtime values cannot be referenced in patterns

mir_build_unreachable_pattern = unreachable pattern
    .label = unreachable pattern
    .catchall_label = matches any value

mir_build_const_pattern_depends_on_generic_parameter =
    constant pattern depends on a generic parameter

mir_build_could_not_eval_const_pattern = could not evaluate constant pattern

mir_build_lower_range_bound_must_be_less_than_or_equal_to_upper =
    lower range bound must be less than or equal to upper
    .label = lower bound larger than upper bound
    .teach_note = When matching against a range, the compiler verifies that the range is non-empty. Range patterns include both end-points, so this is equivalent to requiring the start of the range to be less than or equal to the end of the range.

mir_build_literal_in_range_out_of_bounds =
    literal out of range for `{$ty}`
    .label = this value doesn't fit in `{$ty}` whose maximum value is `{$max}`

mir_build_lower_range_bound_must_be_less_than_upper = lower range bound must be less than upper

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

mir_build_bindings_with_variant_name =
    pattern binding `{$ident}` is named the same as one of the variants of the type `{$ty_path}`
    .suggestion = to match on the variant, qualify the path

mir_build_irrefutable_let_patterns_generic_let = irrefutable `let` {$count ->
        [one] pattern
        *[other] patterns
    }
    .note = {$count ->
        [one] this pattern
        *[other] these patterns
    } will always match, so the `let` is useless
    .help = consider removing `let`

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

mir_build_borrow_of_moved_value = borrow of moved value
    .label = value moved into `{$name}` here
    .occurs_because_label = move occurs because `{$name}` has type `{$ty}` which does not implement the `Copy` trait
    .value_borrowed_label = value borrowed here after move
    .suggestion = borrow this binding in the pattern to avoid moving the value

mir_build_multiple_mut_borrows = cannot borrow value as mutable more than once at a time

mir_build_already_borrowed = cannot borrow value as mutable because it is also borrowed as immutable

mir_build_already_mut_borrowed = cannot borrow value as immutable because it is also borrowed as mutable

mir_build_moved_while_borrowed = cannot move out of value because it is borrowed

mir_build_mutable_borrow = value is mutably borrowed by `{$name}` here

mir_build_borrow = value is borrowed by `{$name}` here

mir_build_moved = value is moved into `{$name}` here

mir_build_union_pattern = cannot use unions in constant patterns

mir_build_type_not_structural =
     to use a constant of type `{$non_sm_ty}` in a pattern, `{$non_sm_ty}` must be annotated with `#[derive(PartialEq, Eq)]`

mir_build_unsized_pattern = cannot use unsized non-slice type `{$non_sm_ty}` in constant patterns

mir_build_invalid_pattern = `{$non_sm_ty}` cannot be used in patterns

mir_build_float_pattern = floating-point types cannot be used in patterns

mir_build_pointer_pattern = function pointers and unsized pointers in patterns behave unpredictably and should not be relied upon. See https://github.com/rust-lang/rust/issues/70861 for details.

mir_build_indirect_structural_match =
    to use a constant of type `{$non_sm_ty}` in a pattern, `{$non_sm_ty}` must be annotated with `#[derive(PartialEq, Eq)]`

mir_build_nontrivial_structural_match =
    to use a constant of type `{$non_sm_ty}` in a pattern, the constant's initializer must be trivial or `{$non_sm_ty}` must be annotated with `#[derive(PartialEq, Eq)]`

mir_build_overlapping_range_endpoints = multiple patterns overlap on their endpoints
    .range = ... with this range
    .note = you likely meant to write mutually exclusive ranges

mir_build_non_exhaustive_omitted_pattern = some variants are not matched explicitly
    .help = ensure that all variants are matched explicitly by adding the suggested match arms
    .note = the matched value is of type `{$scrut_ty}` and the `non_exhaustive_omitted_patterns` attribute was found

mir_build_uncovered = {$count ->
        [1] pattern `{$witness_1}`
        [2] patterns `{$witness_1}` and `{$witness_2}`
        [3] patterns `{$witness_1}`, `{$witness_2}` and `{$witness_3}`
        *[other] patterns `{$witness_1}`, `{$witness_2}`, `{$witness_3}` and {$remainder} more
    } not covered

mir_build_pattern_not_covered = refutable pattern in {$origin}
    .pattern_ty = the matched value is of type `{$pattern_ty}`

mir_build_inform_irrefutable = `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with only one variant

mir_build_more_information = for more information, visit https://doc.rust-lang.org/book/ch18-02-refutability.html

mir_build_res_defined_here = {$res} defined here

mir_build_adt_defined_here = `{$ty}` defined here

mir_build_variant_defined_here = not covered

mir_build_interpreted_as_const = introduce a variable instead

mir_build_confused = missing patterns are not covered because `{$variable}` is interpreted as {$article} {$res} pattern, not a new variable

mir_build_suggest_if_let = you might want to use `if let` to ignore the {$count ->
        [one] variant that isn't
        *[other] variants that aren't
    } matched

mir_build_suggest_let_else = you might want to use `let else` to handle the {$count ->
        [one] variant that isn't
        *[other] variants that aren't
    } matched

mir_build_suggest_attempted_int_lit = alternatively, you could prepend the pattern with an underscore to define a new named variable; identifiers cannot begin with digits


mir_build_rustc_box_attribute_error = `#[rustc_box]` attribute used incorrectly
    .attributes = no other attributes may be applied
    .not_box = `#[rustc_box]` may only be applied to a `Box::new()` call
    .missing_box = `#[rustc_box]` requires the `owned_box` lang item
