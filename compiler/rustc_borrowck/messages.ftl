borrowck_assign_due_to_use_closure =
    assignment occurs due to use in closure

borrowck_assign_due_to_use_coroutine =
    assign occurs due to use in coroutine

borrowck_assign_part_due_to_use_closure =
    assignment to part occurs due to use in closure

borrowck_assign_part_due_to_use_coroutine =
    assign to part occurs due to use in coroutine

borrowck_borrow_due_to_use_closure =
    borrow occurs due to use in closure

borrowck_borrow_due_to_use_coroutine =
    borrow occurs due to use in coroutine

borrowck_calling_operator_moves =
    calling this operator moves the value

borrowck_calling_operator_moves_lhs =
    calling this operator moves the left-hand side

borrowck_cannot_move_when_borrowed =
    cannot move out of {$place ->
        [value] value
        *[other] {$place}
    } because it is borrowed
    .label = borrow of {$borrow_place ->
        [value] value
        *[other] {$borrow_place}
    } occurs here
    .move_label = move out of {$value_place ->
        [value] value
        *[other] {$value_place}
    } occurs here

borrowck_capture_immute =
    capture is immutable because of use here

borrowck_capture_move =
    capture is moved because of use here

borrowck_capture_mut =
    capture is mutable because of use here

borrowck_closure_inferred_mut = inferred to be a `FnMut` closure

borrowck_closure_invoked_twice =
    closure cannot be invoked more than once because it moves the variable `{$place_name}` out of its environment

borrowck_closure_moved_twice =
    closure cannot be moved more than once as it is not `Copy` due to moving the variable `{$place_name}` out of its environment

borrowck_consider_borrow_type_contents =
    help: consider calling `.as_ref()` or `.as_mut()` to borrow the type's contents

borrowck_could_not_normalize =
    could not normalize `{$value}`

borrowck_could_not_prove =
    could not prove `{$predicate}`

borrowck_dereference_suggestion =
    dereference the return value

borrowck_func_take_self_moved_place =
    `{$func}` takes ownership of the receiver `self`, which moves {$place_name}

borrowck_generic_does_not_live_long_enough =
    `{$kind}` does not live long enough

borrowck_higher_ranked_lifetime_error =
    higher-ranked lifetime error

borrowck_higher_ranked_subtype_error =
    higher-ranked subtype error

borrowck_implicit_static =
    this has an implicit `'static` lifetime requirement

borrowck_implicit_static_introduced =
    calling this method introduces the `impl`'s `'static` requirement

borrowck_implicit_static_relax =
    consider relaxing the implicit `'static` requirement

borrowck_lifetime_constraints_error =
    lifetime may not live long enough

borrowck_limitations_implies_static =
    due to current limitations in the borrow checker, this implies a `'static` lifetime

borrowck_move_closure_suggestion =
    consider adding 'move' keyword before the nested closure

borrowck_move_out_place_here =
    {$place} is moved here

borrowck_move_unsized =
    cannot move a value of type `{$ty}`
    .label = the size of `{$ty}` cannot be statically determined

borrowck_moved_a_fn_once_in_call =
    this value implements `FnOnce`, which causes it to be moved when called

borrowck_moved_a_fn_once_in_call_call =
    `FnOnce` closures can only be called once

borrowck_moved_a_fn_once_in_call_def =
    `{$ty}` is made to be an `FnOnce` closure here

borrowck_moved_due_to_await =
    {$place_name} {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to this {$is_loop_message ->
        [true] await, in previous iteration of loop
        *[false] await
    }

borrowck_moved_due_to_call =
    {$place_name} {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to this {$is_loop_message ->
        [true] call, in previous iteration of loop
        *[false] call
    }

borrowck_moved_due_to_implicit_into_iter_call =
    {$place_name} {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to this implicit call to {$is_loop_message ->
        [true] `.into_iter()`, in previous iteration of loop
        *[false] `.into_iter()`
    }

borrowck_moved_due_to_method_call =
    {$place_name} {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to this method {$is_loop_message ->
        [true] call, in previous iteration of loop
        *[false] call
    }

borrowck_moved_due_to_usage_in_operator =
    {$place_name} {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to usage in {$is_loop_message ->
        [true] operator, in previous iteration of loop
        *[false] operator
    }

borrowck_opaque_type_lifetime_mismatch =
    opaque type used twice with different lifetimes
    .label = lifetime `{$arg}` used here
    .prev_lifetime_label = lifetime `{$prev}` previously used here
    .note = if all non-lifetime generic parameters are the same, but the lifetime parameters differ, it is not possible to differentiate the opaque types

borrowck_partial_var_move_by_use_in_closure =
    variable {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to use in closure

borrowck_partial_var_move_by_use_in_coroutine =
    variable {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to use in coroutine

borrowck_restrict_to_static =
    consider restricting the type parameter to the `'static` lifetime

borrowck_returned_async_block_escaped =
    returns an `async` block that contains a reference to a captured variable, which then escapes the closure body

borrowck_returned_closure_escaped =
    returns a closure that contains a reference to a captured variable, which then escapes the closure body

borrowck_returned_lifetime_short =
    {$category_desc}requires that `{$free_region_name}` must outlive `{$outlived_fr_name}`

borrowck_returned_lifetime_wrong =
    {$mir_def_name} was supposed to return data with lifetime `{$outlived_fr_name}` but it is returning data with lifetime `{$fr_name}`

borrowck_returned_ref_escaped =
    returns a reference to a captured variable which escapes the closure body

borrowck_simd_intrinsic_arg_const =
    {$arg ->
        [1] 1st
        [2] 2nd
        [3] 3rd
        *[other] {$arg}th
    } argument of `{$intrinsic}` is required to be a `const` item

borrowck_suggest_create_fresh_reborrow =
    consider reborrowing the `Pin` instead of moving it

borrowck_suggest_iterate_over_slice =
    consider iterating over a slice of the `{$ty}`'s content to avoid moving into the `for` loop

borrowck_tail_expr_drop_order = relative drop order changing in Rust 2024
    .label = this temporary value will be dropped at the end of the block
    .note = consider using a `let` binding to ensure the value will live long enough

borrowck_ty_no_impl_copy =
    {$is_partial_move ->
        [true] partial move
        *[false] move
    } occurs because {$place} has type `{$ty}`, which does not implement the `Copy` trait

borrowck_use_due_to_use_closure =
    use occurs due to use in closure

borrowck_use_due_to_use_coroutine =
    use occurs due to use in coroutine

borrowck_used_impl_require_static =
    the used `impl` has a `'static` requirement

borrowck_value_capture_here =
    value captured {$is_within ->
        [true] here by coroutine
        *[false] here
    }

borrowck_value_moved_here =
    value {$is_partial ->
        [true] partially moved
        *[false] moved
    } {$is_move_msg ->
        [true] into closure here
        *[false] here
    }{$is_loop_message ->
        [true] , in previous iteration of loop
        *[false] {""}
    }

borrowck_var_borrow_by_use_in_closure =
    borrow occurs due to use in closure

borrowck_var_borrow_by_use_in_coroutine =
    borrow occurs due to use in coroutine

borrowck_var_borrow_by_use_place_in_closure =
    {$is_single_var ->
        *[true] borrow occurs
        [false] borrows occur
    } due to use of {$place} in closure

borrowck_var_borrow_by_use_place_in_coroutine =
    {$is_single_var ->
        *[true] borrow occurs
        [false] borrows occur
    } due to use of {$place} in coroutine

borrowck_var_cannot_escape_closure =
    captured variable cannot escape `FnMut` closure body
    .note = `FnMut` closures only have access to their captured variables while they are executing...
    .cannot_escape = ...therefore, they cannot allow references to captured variables to escape

borrowck_var_does_not_need_mut =
    variable does not need to be mutable
    .suggestion = remove this `mut`

borrowck_var_first_borrow_by_use_place_in_closure =
    first borrow occurs due to use of {$place} in closure

borrowck_var_first_borrow_by_use_place_in_coroutine =
    first borrow occurs due to use of {$place} in coroutine

borrowck_var_here_captured = variable captured here

borrowck_var_here_defined = variable defined here

borrowck_var_move_by_use_in_closure =
    move occurs due to use in closure

borrowck_var_move_by_use_in_coroutine =
    move occurs due to use in coroutine

borrowck_var_mutable_borrow_by_use_place_in_closure =
    mutable borrow occurs due to use of {$place} in closure

borrowck_var_second_borrow_by_use_place_in_closure =
    second borrow occurs due to use of {$place} in closure

borrowck_var_second_borrow_by_use_place_in_coroutine =
    second borrow occurs due to use of {$place} in coroutine
