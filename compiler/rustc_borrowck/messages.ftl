borrowck_move_unsized =
    cannot move a value of type `{$ty}`
    .label = the size of `{$ty}` cannot be statically determined

borrowck_higher_ranked_lifetime_error =
    higher-ranked lifetime error

borrowck_could_not_prove =
    could not prove `{$predicate}`

borrowck_could_not_normalize =
    could not normalize `{$value}`

borrowck_higher_ranked_subtype_error =
    higher-ranked subtype error

borrowck_generic_does_not_live_long_enough =
    `{$kind}` does not live long enough

borrowck_move_borrowed =
    cannot move out of `{$desc}` because it is borrowed

borrowck_var_does_not_need_mut =
    variable does not need to be mutable
    .suggestion = remove this `mut`

borrowck_var_cannot_escape_closure =
    captured variable cannot escape `FnMut` closure body
    .note = `FnMut` closures only have access to their captured variables while they are executing...
    .cannot_escape = ...therefore, they cannot allow references to captured variables to escape

borrowck_var_here_defined = variable defined here

borrowck_var_here_captured = variable captured here

borrowck_closure_inferred_mut = inferred to be a `FnMut` closure

borrowck_returned_closure_escaped =
    returns a closure that contains a reference to a captured variable, which then escapes the closure body

borrowck_returned_async_block_escaped =
    returns an `async` block that contains a reference to a captured variable, which then escapes the closure body

borrowck_returned_ref_escaped =
    returns a reference to a captured variable which escapes the closure body

borrowck_lifetime_constraints_error =
    lifetime may not live long enough

borrowck_returned_lifetime_wrong =
    {$mir_def_name} was supposed to return data with lifetime `{$outlived_fr_name}` but it is returning data with lifetime `{$fr_name}`

borrowck_returned_lifetime_short =
    {$category_desc}requires that `{$free_region_name}` must outlive `{$outlived_fr_name}`

borrowck_used_impl_require_static =
    the used `impl` has a `'static` requirement

borrowck_borrow_due_to_use_generator =
    borrow occurs due to use in generator

borrowck_use_due_to_use_generator =
    use occurs due to use in generator

borrowck_assign_due_to_use_generator =
    assign occurs due to use in generator

borrowck_assign_part_due_to_use_generator =
    assign to part occurs due to use in generator

borrowck_borrow_due_to_use_closure =
    borrow occurs due to use in closure

borrowck_use_due_to_use_closure =
    use occurs due to use in closure

borrowck_assign_due_to_use_closure =
    assignment occurs due to use in closure

borrowck_assign_part_due_to_use_closure =
    assignment to part occurs due to use in closure

borrowck_capture_immute =
    capture is immutable because of use here

borrowck_capture_mut =
    capture is mutable because of use here

borrowck_capture_move =
    capture is moved because of use here

borrowck_var_borrow_by_use_place_in_generator =
    {$is_single_var ->
        *[true] borrow occurs
        [false] borrows occur
    } due to use of {$place} in generator

borrowck_var_borrow_by_use_place_in_closure =
    {$is_single_var ->
        *[true] borrow occurs
        [false] borrows occur
    } due to use of {$place} in closure

borrowck_var_borrow_by_use_in_generator =
    borrow occurs due to use in generator

borrowck_var_borrow_by_use_in_closure =
    borrow occurs due to use in closure

borrowck_var_move_by_use_place_in_generator =
    move occurs due to use of {$place} in generator

borrowck_var_move_by_use_place_in_closure =
    move occurs due to use of {$place} in closure

borrowck_var_move_by_use_in_generator =
    move occurs due to use in generator

borrowck_var_move_by_use_in_closure =
    move occurs due to use in closure

borrowck_partial_var_move_by_use_in_generator =
    variable {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to use in generator

borrowck_partial_var_move_by_use_in_closure =
    variable {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to use in closure

borrowck_var_first_borrow_by_use_place_in_generator =
    first borrow occurs due to use of {$place} in generator

borrowck_var_first_borrow_by_use_place_in_closure =
    first borrow occurs due to use of {$place} in closure

borrowck_var_second_borrow_by_use_place_in_generator =
    second borrow occurs due to use of {$place} in generator

borrowck_var_second_borrow_by_use_place_in_closure =
    second borrow occurs due to use of {$place} in closure

borrowck_var_mutable_borrow_by_use_place_in_closure =
    mutable borrow occurs due to use of {$place} in closure

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

borrowck_opaque_type_non_generic_param =
    expected generic {$kind} parameter, found `{$ty}`
    .label = {STREQ($ty, "'static") ->
        [true] cannot use static lifetime; use a bound lifetime instead or remove the lifetime parameter from the opaque type
        *[other] this generic parameter must be used with a generic {$kind} parameter
    }

borrowck_moved_due_to_call =
    {$place_name} {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to this {$is_loop_message ->
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

borrowck_moved_due_to_await =
    {$place_name} {$is_partial ->
        [true] partially moved
        *[false] moved
    } due to this {$is_loop_message ->
        [true] await, in previous iteration of loop
        *[false] await
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

borrowck_consider_borrow_type_contents =
    help: consider calling `.as_ref()` or `.as_mut()` to borrow the type's contents

borrowck_moved_a_fn_once_in_call =
    this value implements `FnOnce`, which causes it to be moved when called

borrowck_calling_operator_moves_lhs =
    calling this operator moves the left-hand side

borrowck_func_take_self_moved_place =
    `{$func}` takes ownership of the receiver `self`, which moves {$place_name}

borrowck_suggest_iterate_over_slice =
    consider iterating over a slice of the `{$ty}`'s content to avoid moving into the `for` loop

borrowck_suggest_create_freash_reborrow =
    consider reborrowing the `Pin` instead of moving it

borrowck_value_capture_here =
    value captured {$is_within ->
        [true] here by generator
        *[false] here
    }

borrowck_move_out_place_here =
    {$place} is moved here

borrowck_closure_invoked_twice =
    closure cannot be invoked more than once because it moves the variable `{$place_name}` out of its environment

borrowck_closure_moved_twice =
    closure cannot be moved more than once as it is not `Copy` due to moving the variable `{$place_name}` out of its environment

borrowck_ty_no_impl_copy =
    {$is_partial_move ->
        [true] partial move
        *[false] move
    } occurs because {$place} has type `{$ty}`, which does not implement the `Copy` trait
