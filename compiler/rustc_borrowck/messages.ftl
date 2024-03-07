borrowck_and_data_moved_here = ...and here

borrowck_assign_data_behind_const_pointer =
    cannot assign to data in a `*const` pointer

borrowck_assign_data_behind_deref =
    cannot assign to data in {$name}

borrowck_assign_data_behind_index =
    cannot assign to data in an index of {$name}

borrowck_assign_data_behind_ref =
    cannot assign to data in a `&` reference

borrowck_assign_due_to_use_closure =
    assignment occurs due to use in closure

borrowck_assign_due_to_use_coroutine =
    assign occurs due to use in coroutine

borrowck_assign_part_due_to_use_closure =
    assignment to part occurs due to use in closure

borrowck_assign_part_due_to_use_coroutine =
    assign to part occurs due to use in coroutine

borrowck_assign_place_behind_const_pointer =
    cannot assign to `{$place}`, which is behind a `*const` pointer

borrowck_assign_place_behind_deref =
    cannot assign to `{$place}`, which is behind {$name}

borrowck_assign_place_behind_index =
    cannot assign to `{$place}`, which is behind an index of {$ty}

borrowck_assign_place_behind_ref =
    cannot assign to `{$place}`, which is behind a `&` reference

borrowck_assign_place_declared_immute =
    cannot assign to `{$place}`, as it is not declared as mutable

borrowck_assign_place_in_fn =
    cannot assign to `{$place}`, as it is a captured variable in a `Fn` closure

borrowck_assign_place_in_pattern_guard_immute =
    cannot assign to `{$place}`, as it is immutable for the pattern guard

borrowck_assign_place_static =
    cannot assign to immutable static item `{$place}`

borrowck_assign_symbol_declared_immute =
    cannot assign to `{$place}`, as `{$name}` is not declared as mutable

borrowck_assign_symbol_static =
    cannot assign to `{$place}`, as `{$static_name}` is an immutable static item

borrowck_assign_upvar_in_fn =
    cannot assign to `{$place}`, as `Fn` closures cannot mutate their captured variables

borrowck_borrow_due_to_use_closure =
    borrow occurs due to use in closure

borrowck_borrow_due_to_use_coroutine =
    borrow occurs due to use in coroutine

borrowck_borrow_occurs_here = {$kind} borrow occurs here

borrowck_borrow_occurs_here_overlap =
    {$kind_new} borrow of {$msg_new} -- which overlaps with {$msg_old} -- occurs here

borrowck_borrow_occurs_here_via =
    {$kind_old} borrow occurs {$is_msg_old_empty ->
        *[true] here
        [false] here (via {$msg_old})
    }

borrowck_borrowed_temporary_value_dropped =
    temporary value dropped while borrowed

borrowck_calling_operator_moves_lhs =
    calling this operator moves the left-hand side

borrowck_cannot_assign =
    cannot assign

borrowck_cannot_assign_to_borrowed =
    cannot assign to {$desc ->
        [value] value
        *[other] {$desc}
    } because it is borrowed
    .label = {$desc ->
        [value] value
        *[other] {$desc}
    } is assigned to here but it was already borrowed
    .borrow_here_label = {$desc ->
        [value] value
        *[other] {$desc}
    } is borrowed here

borrowck_cannot_borrow_across_coroutine_yield =
    borrow may still be in use when {$coroutine_kind} yields
    .label = possible yield occurs here

borrowck_cannot_borrow_across_destructor =
    borrow may still be in use when destructor runs

borrowck_cannot_borrow_mut =
    cannot borrow as mutable

borrowck_cannot_move_out_of_interior_of_drop =
    cannot move out of type `{$container_ty}`, which implements the `Drop` trait
    .label = cannot move out of here

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

borrowck_cannot_mutably_borrow_multiply =
    cannot borrow {$is_place_empty ->
        *[true] {$new_place_name}
        [false] {$new_place_name} (via {$place})
    } as mutable more than once at a time
    .label = first mutable borrow occurs {$is_old_place_empty ->
        *[true] here
        [false] here (via {$old_place})
    }
    .second_mut_borrow_label = second mutable borrow occurs {$is_place_empty ->
        *[true] here
        [false] here (via {$place})
    }
    .first_mut_end_label = first borrow ends here

borrowck_cannot_mutably_borrow_multiply_same_span =
    cannot borrow {$is_place_empty ->
        *[true] {$new_place_name}
        [false] {$new_place_name} (via {$place})
    } as mutable more than once at a time
    .label = mutable borrow ends here

borrowck_cannot_reassign_immutable_arg =
    cannot assign to immutable argument {$place}

borrowck_cannot_reassign_immutable_var =
    cannot assign twice to immutable variable {$place}

borrowck_cannot_reborrow_already_borrowed =
    cannot borrow {$is_msg_new_empty ->
        *[true] {$desc_new}
        [false] {$desc_new} (via {$msg_new})
    } as {$kind_new} because {$noun_old} is also borrowed as {$is_msg_old_empty ->
        *[true] {$kind_old}
        [false] {$kind_old} (via {$msg_old})
    }
    .label = {$kind_old} borrow ends here

borrowck_cannot_reborrow_already_uniquely_borrowed =
    cannot borrow {$desc_new}{$opt_via} as {$kind_new} because previous closure requires unique access
    .label = {$second_borrow_desc}borrow occurs here{$opt_via}
    .occurs_label = {$container_name} construction occurs here{$old_opt_via}
    .ends_label = borrow from closure ends here

borrowck_cannot_return_ref_to_local =
    cannot {$return_kind} {$reference} {$local}
    .label = {$return_kind}s a {$reference} data owned by the current function

borrowck_cannot_uniquely_borrow_by_one_closure =
    closure requires unique access to {$desc_new} but {$noun_old} is already borrowed{$old_opt_via}
    .label = {$container_name} construction occurs here{$opt_via}
    .occurs_label = borrow occurs here{$old_opt_via}
    .ends_label = borrow ends here

borrowck_cannot_uniquely_borrow_by_two_closures =
    two closures require unique access to {$desc ->
        [value] value
        *[other] {$desc}
    } at the same time
    .label = borrow from first closure ends here
    .new_span_label = second closure is constructed here

borrowck_cannot_use_static_lifetime_here =
    cannot use static lifetime; use a bound lifetime instead or remove the lifetime parameter from the opaque type

borrowck_cannot_use_when_mutably_borrowed =
    cannot use {$desc ->
        [value] value
        *[other] {$desc}
    } because it was mutably borrowed
    .label = use of borrowed {$borrow_desc ->
        [value] value
        *[other] {$borrow_desc}
    }
    .borrow_span_label = {$borrow_desc ->
        [value] value
        *[other] {$borrow_desc}
    } is borrowed here

borrowck_capture_immute =
    capture is immutable because of use here

borrowck_capture_move =
    capture is moved because of use here

borrowck_capture_mut =
    capture is mutable because of use here

borrowck_closure_borrowing_outlive_function =
    {$closure_kind} may outlive the current function, but it borrows {$borrowed_path}, which is owned by the current function
    .label = may outlive borrowed value {$borrowed_path}
    .path_label = {$borrowed_path} is borrowed here

borrowck_closure_cannot_invoke_again =
    closure cannot be invoked more than once because it moves the variable `{$place}` out of its environment

borrowck_closure_cannot_move_again =
    closure cannot be moved more than once as it is not `Copy` due to moving the variable `{$place}` out of its environment

borrowck_closure_inferred_mut = inferred to be a `FnMut` closure

borrowck_closure_invoked_twice =
    closure cannot be invoked more than once because it moves the variable `{$place_name}` out of its environment

borrowck_closure_moved_twice =
    closure cannot be moved more than once as it is not `Copy` due to moving the variable `{$place_name}` out of its environment

borrowck_closures_constructed_here =
    closures are constructed here in different iterations of loop

borrowck_consider_add_lifetime_bound =
    consider adding the following bound: `{$fr_name}: {$outlived_fr_name}`

borrowck_consider_add_semicolon =
    consider adding semicolon after the expression so its temporaries are dropped sooner, before the local variables declared by the block are dropped

borrowck_consider_borrow_type_contents =
    help: consider calling `.as_ref()` or `.as_mut()` to borrow the type's contents

borrowck_consider_forcing_temporary_drop_sooner =
    the temporary is part of an expression at the end of a block;
    consider forcing this temporary to be dropped sooner, before the block's local variables are dropped

borrowck_consider_move_expression_end_of_block =
    for example, you could save the expression's value in a new local variable `x` and then make `x` be the expression at the end of the block

borrowck_could_not_normalize =
    could not normalize `{$value}`

borrowck_could_not_prove =
    could not prove `{$predicate}`

borrowck_data_moved_here =
    data moved here

borrowck_define_const_type =
    defining constant type: {$type_name}

borrowck_define_inline_constant_type =
    defining inline constant type: {$type_name}

borrowck_define_type =
    defining type: {$type_name}

borrowck_define_type_with_closure_args =
    defining type: {$type_name} with closure args [
    {"    "}{$subsets},
    ]

borrowck_define_type_with_generator_args =
    defining type: {$type_name} with generator args [
    {"    "}{$subsets},
    ]

borrowck_empty_label = {""}

borrowck_expects_fn_not_fnmut =
    expects `Fn` instead of `FnMut`

borrowck_expects_fnmut_not_fn =
    change this to accept `FnMut` instead of `Fn`

borrowck_first_closure_constructed_here =
    first closure is constructed here

borrowck_func_take_self_moved_place =
    `{$func}` takes ownership of the receiver `self`, which moves {$place_name}

borrowck_function_takes_self_ownership =
    this function takes ownership of the receiver `self`, which moves {$place_name}

borrowck_generic_does_not_live_long_enough =
    `{$kind}` does not live long enough

borrowck_higher_ranked_lifetime_error =
    higher-ranked lifetime error

borrowck_higher_ranked_subtype_error =
    higher-ranked subtype error

borrowck_in_this_closure =
    in this closure

borrowck_lifetime_appears_here_in_impl =
    lifetime `{$rg_name}` appears in the `impl`'s {$location}

borrowck_lifetime_appears_in_type =
    lifetime `{$rg_name}` appears in the type {$type_name}

borrowck_lifetime_appears_in_type_of =
    lifetime `{$rg_name}` appears in the type of `{$upvar_name}`

borrowck_lifetime_constraints_error =
    lifetime may not live long enough

borrowck_modify_ty_methods_help =
    to modify a `{$ty}`, use `.get_mut()`, `.insert()` or the entry API

borrowck_move_borrowed =
    cannot move out of `{$desc}` because it is borrowed

borrowck_move_out_place_here =
    {$place} is moved here

borrowck_move_unsized =
    cannot move a value of type `{$ty}`
    .label = the size of `{$ty}` cannot be statically determined

borrowck_moved_a_fn_once_in_call =
    this value implements `FnOnce`, which causes it to be moved when called

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

borrowck_moved_var_cannot_copy =
    move occurs because these variables have types that don't implement the `Copy` trait

borrowck_mut_borrow_data_behind_const_pointer =
    cannot borrow data in a `*const` pointer as mutable

borrowck_mut_borrow_data_behind_deref =
    cannot borrow data in {$name} as mutable

borrowck_mut_borrow_data_behind_index =
    cannot borrow data in an index of {$name} as mutable

borrowck_mut_borrow_data_behind_ref =
    cannot borrow data in a `&` reference as mutable

borrowck_mut_borrow_place_declared_immute =
    cannot borrow `{$place}` as mutable, as it is not declared as mutable

borrowck_mut_borrow_place_in_pattern_guard_immute =
    cannot borrow `{$place}` as mutable, as it is immutable for the pattern guard

borrowck_mut_borrow_place_static =
    cannot borrow immutable static item `{$place}` as mutable

borrowck_mut_borrow_self_behind_const_pointer =
    cannot borrow `{$place}` as mutable, as it is behind a `*const` pointer

borrowck_mut_borrow_self_behind_deref =
    cannot borrow `{$place}` as mutable, as it is behind {$name}

borrowck_mut_borrow_self_behind_index =
    cannot borrow `{$place}` as mutable, as it is behind an index of {$name}

borrowck_mut_borrow_self_behind_ref =
    cannot borrow `{$place}` as mutable, as it is behind a `&` reference

borrowck_mut_borrow_self_in_fn =
    cannot borrow `{$place}` as mutable, as it is a captured variable in a `Fn` closure

borrowck_mut_borrow_symbol_declared_immute =
    cannot borrow `{$place}` as mutable, as `{$name}` is not declared as mutable

borrowck_mut_borrow_symbol_static =
    cannot borrow `{$place}` as mutable, as `{$static_name}` is an immutable static item

borrowck_mut_borrow_upvar_in_fn =
    cannot borrow `{$place}` as mutable, as `Fn` closures cannot mutate their captured variables

borrowck_mutably_borrow_multiply_loop_label =
    {$is_place_empty ->
        *[true] {$new_place_name}
        [false] {$new_place_name} (via {$place})
    } was mutably borrowed here in the previous iteration of the loop{$place}

borrowck_name_this_region =
    let's call this `{$rg_name}`

borrowck_non_defining_opaque_type =
    non-defining opaque type use in defining scope

borrowck_opaque_type_non_generic_param =
    expected generic {$kind} parameter, found `{$ty}`
    .label = {STREQ($ty, "'static") ->
        [true] cannot use static lifetime; use a bound lifetime instead or remove the lifetime parameter from the opaque type
        *[other] this generic parameter must be used with a generic {$kind} parameter
    }

borrowck_outlive_constraint_need_borrow_for =
    {$category}requires that {$desc} is borrowed for `{$region_name}`

borrowck_outlive_constraint_need_borrow_lasts =
    {$category}requires that `{$borrow_desc}` lasts for `{$region_name}`

borrowck_outlive_constraint_need_borrow_lasts_for =
    {$category}requires that {$borrow_desc} is borrowed for `{$region_name}`

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

borrowck_path_does_not_live_long_enough =
    {$path} does not live long enough

borrowck_perhaps_save_in_new_local_to_drop =
    for example, you could save the expression's value in a new local variable `x` and then make `x` be the expression at the end of the block

borrowck_return_fnmut =
    change this to return `FnMut` instead of `Fn`

borrowck_returned_async_block_escaped =
    returns an `async` block that contains a reference to a captured variable, which then escapes the closure body

borrowck_returned_closure_escaped =
    returns a closure that contains a reference to a captured variable, which then escapes the closure body

borrowck_returned_lifetime_short =
    {$category}requires that `{$free_region_name}` must outlive `{$outlived_fr_name}`

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

borrowck_suggest_create_freash_reborrow =
    consider reborrowing the `Pin` instead of moving it

borrowck_suggest_iterate_over_slice =
    consider iterating over a slice of the `{$ty}`'s content to avoid moving into the `for` loop

borrowck_thread_local_outlive_function =
    thread-local variable borrowed past end of function

borrowck_ty_no_impl_copy =
    {$is_partial_move ->
        [true] partial move
        *[false] move
    } occurs because {$place} has type `{$ty}`, which does not implement the `Copy` trait

borrowck_type_parameter_not_used_in_trait_type_alias =
    type parameter `{$ty}` is part of concrete type but not used in parameter list for the `impl Trait` type alias

borrowck_upvar_need_mut_due_to_borrow =
    calling `{$place}` requires mutable binding due to mutable borrow of `{$upvar}`

borrowck_upvar_need_mut_due_to_mutation =
    calling `{$place}` requires mutable binding due to possible mutation of `{$upvar}`

borrowck_use_due_to_use_closure =
    use occurs due to use in closure

borrowck_use_due_to_use_coroutine =
    use occurs due to use in coroutine

borrowck_used_here_by_closure =
    used here by closure

borrowck_used_impl_require_static =
    the used `impl` has a `'static` requirement

borrowck_value_capture_here =
    value captured {$is_within ->
        [true] here by coroutine
        *[false] here
    }

borrowck_value_moved_here =
    {$is_partial ->
        [true] value partially
        *[false] value
    } {$is_move_msg ->
        [true] moved into closure
        *[false] moved
    } {$is_loop_message ->
        [true] here, in previous iteration of loop
        *[false] here
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

borrowck_yield_type_is_type =
    yield type is {$type_name}
